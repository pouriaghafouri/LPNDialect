#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

using GuardSet = SmallVector<unsigned, 4>;
using GuardStateVector = SmallVector<GuardSet, 4>;

static void insertGuard(GuardSet &set, unsigned guardId) {
  auto it = llvm::lower_bound(set, guardId);
  if (it == set.end() || *it != guardId)
    set.insert(it, guardId);
}

static void dedupGuardStates(GuardStateVector &states) {
  llvm::sort(states, [](const GuardSet &a, const GuardSet &b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(),
                                        b.end());
  });
  auto newEnd = std::unique(states.begin(), states.end(),
                            [](const GuardSet &lhs, const GuardSet &rhs) {
                              return lhs == rhs;
                            });
  states.erase(newEnd, states.end());
}

using EmitGuardMap =
    DenseMap<Operation *, SmallVector<GuardSet, 4>>;

static GuardStateVector walkBlock(Block &block, GuardStateVector states,
                                  DenseMap<Value, unsigned> &takeIds,
                                  EmitGuardMap &emitGuards);

static bool mergeGuardStateVectors(GuardStateVector &dest,
                                   const GuardStateVector &source) {
  bool changed = false;
  for (const GuardSet &set : source) {
    bool exists = llvm::any_of(dest, [&](const GuardSet &other) {
      return other == set;
    });
    if (!exists) {
      dest.push_back(set);
      changed = true;
    }
  }
  if (changed)
    dedupGuardStates(dest);
  return changed;
}

static GuardStateVector walkRegion(Region &region, GuardStateVector states,
                                   DenseMap<Value, unsigned> &takeIds,
                                   EmitGuardMap &emitGuards) {
  if (region.empty())
    return states;
  Block &entry = region.front();
  DenseMap<Block *, GuardStateVector> incoming;
  SmallVector<Block *, 4> worklist;
  incoming[&entry] = std::move(states);
  worklist.push_back(&entry);
  GuardStateVector exitStates;

  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    GuardStateVector blockStates = incoming[block];
    GuardStateVector outgoing =
        walkBlock(*block, std::move(blockStates), takeIds, emitGuards);
    Operation *terminator = block->getTerminator();
    if (terminator->getNumSuccessors() == 0) {
      mergeGuardStateVectors(exitStates, outgoing);
      continue;
    }
    for (unsigned idx = 0, e = terminator->getNumSuccessors(); idx < e; ++idx) {
      Block *succ = terminator->getSuccessor(idx);
      GuardStateVector &succStates = incoming[succ];
      if (mergeGuardStateVectors(succStates, outgoing))
        worklist.push_back(succ);
    }
  }

  return exitStates.empty() ? GuardStateVector{} : exitStates;
}

static GuardStateVector
walkChoice(ChoiceOp choice, const GuardStateVector &incoming,
           DenseMap<Value, unsigned> &takeIds, EmitGuardMap &emitGuards) {
  GuardStateVector thenStates =
      walkRegion(choice.getThenRegion(), incoming, takeIds, emitGuards);
  GuardStateVector elseStates =
      walkRegion(choice.getElseRegion(), incoming, takeIds, emitGuards);
  GuardStateVector merged = std::move(thenStates);
  merged.append(elseStates.begin(), elseStates.end());
  dedupGuardStates(merged);
  return merged;
}

static GuardStateVector
walkIf(scf::IfOp ifOp, const GuardStateVector &incoming,
       DenseMap<Value, unsigned> &takeIds, EmitGuardMap &emitGuards) {
  GuardStateVector thenStates =
      walkRegion(ifOp.getThenRegion(), incoming, takeIds, emitGuards);
  GuardStateVector elseStates;
  if (!ifOp.getElseRegion().empty())
    elseStates =
        walkRegion(ifOp.getElseRegion(), incoming, takeIds, emitGuards);
  else
    elseStates = incoming;
  GuardStateVector merged = std::move(thenStates);
  merged.append(elseStates.begin(), elseStates.end());
  dedupGuardStates(merged);
  return merged;
}

static GuardStateVector walkBlock(Block &block, GuardStateVector states,
                                  DenseMap<Value, unsigned> &takeIds,
                                  EmitGuardMap &emitGuards) {
  if (states.empty())
    states.push_back(GuardSet{});

  for (Operation &op : block.without_terminator()) {
    if (auto take = dyn_cast<TakeOp>(&op)) {
      auto idIt = takeIds.find(take.getResult());
      if (idIt == takeIds.end())
        continue;
      for (GuardSet &set : states)
        insertGuard(set, idIt->second);
      continue;
    }
    if (auto emit = dyn_cast<EmitOp>(&op)) {
      auto &entries = emitGuards[emit.getOperation()];
      for (const GuardSet &set : states)
        entries.push_back(set);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      states = walkIf(ifOp, states, takeIds, emitGuards);
      continue;
    }
    if (auto choice = dyn_cast<ChoiceOp>(&op)) {
      states = walkChoice(choice, states, takeIds, emitGuards);
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      (void)walkRegion(forOp.getRegion(), states, takeIds, emitGuards);
      continue;
    }
    if (op.getNumRegions()) {
      for (Region &region : op.getRegions())
        (void)walkRegion(region, states, takeIds, emitGuards);
      continue;
    }
  }

  return states;
}

class LPNSynthesizeGuardPass
    : public PassWrapper<LPNSynthesizeGuardPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNSynthesizeGuardPass)

  StringRef getArgument() const override { return "lpn-synthesize-guards"; }
  StringRef getDescription() const override {
    return "Compute per-transition guard metadata for retain passes.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](TransitionOp trans) {
      if (failed(processTransition(trans)))
        signalPassFailure();
    });
  }

private:
  LogicalResult processTransition(TransitionOp trans) {
    DenseMap<Value, unsigned> takeIds;
    unsigned nextGuardId = 0;
    OpBuilder builder(trans.getContext());
    for (TakeOp take : trans.getBody().getOps<TakeOp>()) {
      unsigned guardId = nextGuardId++;
      take->setAttr(kGuardIdAttr, builder.getI64IntegerAttr(guardId));
      takeIds[take.getResult()] = guardId;
    }

    EmitGuardMap emitGuards;
    GuardStateVector initialStates;
    initialStates.push_back(GuardSet{});
    (void)walkRegion(trans.getBody(), std::move(initialStates), takeIds,
                     emitGuards);

    for (auto &entry : emitGuards) {
      SmallVector<GuardSet, 4> sets = std::move(entry.second);
      dedupGuardStates(sets);
      SmallVector<Attribute, 4> attrSets;
      for (GuardSet &set : sets) {
        SmallVector<Attribute, 4> ints;
        ints.reserve(set.size());
        for (unsigned guardId : set)
          ints.push_back(builder.getI64IntegerAttr(guardId));
        attrSets.push_back(builder.getArrayAttr(ints));
      }
      entry.first->setAttr(kGuardPathsAttr, builder.getArrayAttr(attrSets));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNSynthesizeGuardPass() {
  return std::make_unique<LPNSynthesizeGuardPass>();
}

} // namespace mlir::lpn
