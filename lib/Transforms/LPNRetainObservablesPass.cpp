#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <iterator>

namespace mlir::lpn {
namespace {

//===----------------------------------------------------------------------===//
// Pass overview
//===----------------------------------------------------------------------===//
//
// The retain-observable pass collapses an LPN network down to only the user
// marked observable places.  Instead of relying on structural heuristics (for
// example, "single producer/single consumer" places), the new implementation
// materializes full SSA slices for the token that flows between observables:
//
//   1. For every `lpn.take` result we enumerate all reachable `lpn.emit`
//      operations by walking the SSA use-def chain.  Each take/emit pair
//      becomes an EdgeTemplate that records the source place, the emit target
//      (possibly guarded by token metadata), and the SSA values necessary to
//      rebuild the token/delay/handle expressions later.
//
//   2. The EdgeTemplates form a graph keyed by place symbols.  We DFS through
//      this graph to enumerate every path that starts at an observable place
//      and ends at a different observable place.  Paths may cross many
//      intermediate places/transitions; cycles are avoided via a visited set.
//
//   3. For each observable root with at least one outgoing path we synthesize a
//      new transition.  The transition takes from the root place once, then
//      replays every path by cloning the SSA slices recorded in the templates.
//      Guards become `scf.if` predicates, token edits are faithfully replayed,
//      and per-edge delays are accumulated so the observable-to-observable arc
//      models the same scheduling latency as the original network.
//
// The key data structures are intentionally simple:
//
//   * TokenGuard models a single equality predicate derived from token log
//     metadata.  For example, when a switch indexes into a place_list using the
//     expression `index_cast(token.get \"dst\" - 4)`, each potential egress
//     slot contributes a guard `dst == 4 + slot`.  These guards become the
//     runtime conditions inside synthesized `scf.if` operations.
//
//   * TargetInfo couples a destination place symbol with zero or more guards.
//     This plays the role of the old "ResolvedPlace": it describes a concrete
//     observable/non-observable place that can appear on the path graph while
//     still remembering the predicates that routed the token there.
//
//   * EdgeTemplate is an SSA stencil for a single transition edge.  It stores
//     the SSA value produced by the `lpn.take`, the Value that flowed into the
//     matched `lpn.emit`, the place handle SSA value, and the computed delay.
//     When we later clone the template we simply remap the original take result
//     to the token currently flowing through the synthesized transition.
//
//   * EdgePath is an ordered list of EdgeTemplates that connects two
//     observables.  Emitting the path means running through each template in
//     order, cloning the necessary SSA graph, summing all per-edge delays, and
//     finally emitting into the destination observable place.
//
// This slice-based approach makes the pass agnostic to how many tokens a
// transition consumes or how complex the control-flow/arithmetics between
// takes and emits may be.  As long as the SSA dependencies consist of pure
// ops (arith, token.{get,set,clone}, place utilities, etc.) the pass can clone
// the necessary computation verbatim.  Transitions that require additional
// `lpn.take` operations to produce an observable emit are rejected, because the
// collapsed model would need those tokens as well; this limitation mirrors the
// semantic requirement that observable behavior depends only on observable
// tokens.
//
// Example (simplified from the network benchmark):
//
//   ```
//   %ctrl = lpn.take %tor_ctrl
//   %pkt  = lpn.take %ingress_7
//   %dst  = token.get %pkt, \"dst\"
//   %rel  = arith.subi %dst, %const
//   %idx  = arith.index_cast %rel : i64 to index
//   %list = lpn.place_list {places = [@egress_0, @egress_1]}
//   %tgt  = lpn.place_list.get %list, %idx
//   token.set %pkt, \"hops\", 1
//   lpn.emit %tgt, %pkt, %delay
//   ```
//
// The pass produces two EdgeTemplates (one for each egress slot).  Both store
// the SSA handles for `%pkt`, `%tgt`, and `%delay`, but the TargetInfo differs:
// slot 0 carries a guard `dst == const + 0`, while slot 1 carries
// `dst == const + 1`.  During synthesis the guard becomes an `scf.if` that
// wraps the cloned SSA slice, guaranteeing that only the correct observable
// endpoint fires for any particular token.
//
//===----------------------------------------------------------------------===//

/// Utilities to simplify redundant lpn.choice ladders.
static bool regionIsTriviallyEmpty(Region &region) {
  if (region.empty())
    return true;
  return region.front().getOperations().size() == 1;
}

static ChoiceOp getSingletonChoice(Region &region) {
  if (!region.hasOneBlock())
    return nullptr;
  auto range = region.front().without_terminator();
  if (!llvm::hasSingleElement(range))
    return nullptr;
  return dyn_cast<ChoiceOp>(&*range.begin());
}

static bool spliceBranchIntoParent(Region &from, Block &destBlock) {
  if (from.empty())
    return false;
  Block &srcBlock = from.front();
  auto &srcOps = srcBlock.getOperations();
  if (srcOps.empty())
    return false;
  auto srcBegin = srcOps.begin();
  auto srcEnd = srcOps.end();
  --srcEnd;
  if (srcBegin == srcEnd)
    return false;
  auto &destOps = destBlock.getOperations();
  auto insertIt = destBlock.getTerminator()->getIterator();
  destOps.splice(insertIt, srcOps, srcBegin, srcEnd);
  return true;
}

static bool simplifyChoiceOnce(ChoiceOp op) {
  bool thenEmpty = regionIsTriviallyEmpty(op.getThenRegion());
  bool elseEmpty = regionIsTriviallyEmpty(op.getElseRegion());
  if (thenEmpty && elseEmpty) {
    op.erase();
    return true;
  }
  if (thenEmpty == elseEmpty)
    return false;

  Region &candidateRegion = thenEmpty ? op.getElseRegion() : op.getThenRegion();
  ChoiceOp inner = getSingletonChoice(candidateRegion);
  if (!inner)
    return false;

  bool innerThenEmpty = regionIsTriviallyEmpty(inner.getThenRegion());
  bool innerElseEmpty = regionIsTriviallyEmpty(inner.getElseRegion());
  if (innerThenEmpty == innerElseEmpty)
    return false;

  Region &nonEmptyRegion =
      innerThenEmpty ? inner.getElseRegion() : inner.getThenRegion();
  Block &destBlock = candidateRegion.front();
  if (!spliceBranchIntoParent(nonEmptyRegion, destBlock))
    return false;
  inner.erase();
  return true;
}

static void simplifyChoiceLadders(NetOp net) {
  bool changed = true;
  while (changed) {
    changed = false;
    net.walk([&](ChoiceOp op) -> WalkResult {
      if (simplifyChoiceOnce(op)) {
        changed = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
}

/// Single equality predicate derived from token metadata.
struct TokenGuard {
  Value key;
  llvm::hash_code keyHash = {};
  int64_t equalsValue = 0;
};

/// Destination place plus optional guards describing the routing condition.
struct TargetInfo {
  StringAttr symbol;
  SmallVector<TokenGuard> guards;
};

struct TokenEditSignature {
  llvm::hash_code keyHash = {};
  llvm::hash_code valueHash = {};
};

struct ControlContext {
  Operation *op;
  bool isThen;
};

/// SSA stencil for a single take/emit pair.
struct EdgeTemplate {
  StringAttr source;
  TargetInfo target;
  Value takeValue;
  Value tokenValue;
  Value delayValue;
  SmallVector<ControlContext> contexts;
  SmallVector<TokenEditSignature> editSummary;
};

/// Observable-to-observable chain of edges.
using EdgePath = SmallVector<const EdgeTemplate *>;

struct PathCursor {
  const EdgePath *path;
  size_t edgeIndex;
  size_t contextIndex;
};

static bool equivalentTemplate(const EdgeTemplate *lhs,
                               const EdgeTemplate *rhs) {
  if (lhs == rhs)
    return true;
  if (lhs->target.symbol != rhs->target.symbol)
    return false;
  if (lhs->target.guards.size() != rhs->target.guards.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs->target.guards, rhs->target.guards)) {
    if (a.equalsValue != b.equalsValue)
      return false;
    bool lhsHasKey = static_cast<bool>(a.key);
    bool rhsHasKey = static_cast<bool>(b.key);
    if (lhsHasKey != rhsHasKey)
      return false;
    if (lhsHasKey && a.keyHash != b.keyHash)
      return false;
  }
  if (lhs->editSummary.size() != rhs->editSummary.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs->editSummary, rhs->editSummary)) {
    if (a.keyHash != b.keyHash || a.valueHash != b.valueHash)
      return false;
  }
  return true;
}

static bool equivalentPath(const EdgePath &lhs, const EdgePath &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (!equivalentTemplate(a, b))
      return false;
  return true;
}

static void dedupPaths(
    DenseMap<StringAttr, SmallVector<EdgePath>> &observablePaths) {
  for (auto &entry : observablePaths) {
    SmallVector<EdgePath> unique;
    for (EdgePath &path : entry.second) {
      bool duplicate = false;
      for (const EdgePath &seen : unique)
        if (equivalentPath(path, seen)) {
          duplicate = true;
          break;
        }
      if (!duplicate)
        unique.push_back(path);
    }
    entry.second.swap(unique);
  }
}

// collect all the take dependencies for a value, by going backwards one-by-one through the operations
static void collectTakeDependencies(Value value,
                                    SmallPtrSetImpl<Value> &takes) {
  if (!value)
    return;
  SmallVector<Value, 8> stack;
  stack.push_back(value);
  SmallPtrSet<Value, 16> visited;
  while (!stack.empty()) {
    Value current = stack.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto res = dyn_cast<OpResult>(current)) {
      Operation *def = res.getDefiningOp();
      if (!def)
        continue;
      if (auto take = dyn_cast<TakeOp>(def)) {
        takes.insert(take.getResult());
        continue;
      }
      for (Value operand : def->getOperands())
        stack.push_back(operand);
      continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// Utility helpers
//===----------------------------------------------------------------------===//

static LogicalResult resolvePlaceSymbol(Value handle, StringAttr &symbol) {
  auto ref = handle.getDefiningOp<PlaceRefOp>();
  if (!ref)
    return failure();
  symbol = ref.getPlaceAttr().getAttr();
  return success();
}

static llvm::hash_code hashValueExpr(Value value,
                                     DenseMap<Value, llvm::hash_code> &cache) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    llvm::hash_code h =
        llvm::hash_combine(arg.getArgNumber(),
                           reinterpret_cast<uintptr_t>(arg.getOwner()));
    cache[value] = h;
    return h;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    llvm::hash_code h = llvm::hash_value(value.getAsOpaquePointer());
    cache[value] = h;
    return h;
  }

  unsigned resultNumber = 0;
  if (auto res = dyn_cast<OpResult>(value))
    resultNumber = res.getResultNumber();
  llvm::hash_code h =
      llvm::hash_combine(llvm::hash_value(def->getName().getStringRef()),
                         resultNumber);
  for (NamedAttribute attr : def->getAttrs())
    h = llvm::hash_combine(
        h, llvm::hash_value(attr.getName()),
        llvm::hash_value(attr.getValue().getAsOpaquePointer()));
  for (Value operand : def->getOperands())
    h = llvm::hash_combine(h, hashValueExpr(operand, cache));
  cache[value] = h;
  return h;
}

static llvm::hash_code hashOptionalValue(Value value,
                                         DenseMap<Value, llvm::hash_code> &cache) {
  if (!value)
    return llvm::hash_value(static_cast<void *>(nullptr));
  return hashValueExpr(value, cache);
}

// token edits supported only clone and set ops
// get ops didn't change the token, so they are ignored
static LogicalResult summarizeTokenEdits(
    Value current, Value source, SmallVectorImpl<TokenEditSignature> &edits,
    DenseMap<Value, llvm::hash_code> &hashCache) {
  if (current == source)
    return success();
  Operation *def = current.getDefiningOp();
  if (!def)
    return failure();
  if (auto set = dyn_cast<TokenSetOp>(def)) {
    if (failed(summarizeTokenEdits(set.getToken(), source, edits, hashCache)))
      return failure();
    TokenEditSignature sig;
    sig.keyHash = hashValueExpr(set.getKey(), hashCache);
    sig.valueHash = hashValueExpr(set.getValue(), hashCache);
    edits.push_back(std::move(sig));
    return success();
  }
  if (auto clone = dyn_cast<TokenCloneOp>(def))
    return summarizeTokenEdits(clone.getToken(), source, edits, hashCache);
  return def->emitError("token flow includes unsupported op while summarizing");
}

static Value stripIndexCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<arith::IndexCastOp>())
    current = cast.getIn();
  return current;
}

static std::optional<int64_t> getConstI64(Value value) {
  if (!value)
    return std::nullopt;
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return attr.getValue().getSExtValue();
  }
  return std::nullopt;
}

/// Attempt to infer a metadata guard for a place_list slot.
static std::optional<TokenGuard> matchListIndexGuard(Value index,
                                                     int64_t slot) {
  DenseMap<Value, llvm::hash_code> guardHashCache;
  Value current = stripIndexCasts(index);

  auto buildGuard = [&](TokenGetOp get, int64_t offset,
                        bool add) -> std::optional<TokenGuard> {
    TokenGuard guard;
    guard.key = get.getKey();
    guard.keyHash = hashValueExpr(guard.key, guardHashCache);
    guard.equalsValue = add ? slot - offset : slot + offset;
    return guard;
  };

  if (auto get = current.getDefiningOp<TokenGetOp>())
    return buildGuard(get, /*offset=*/0, /*add=*/true);

  if (auto subi = current.getDefiningOp<arith::SubIOp>()) {
    if (auto lhs = subi.getLhs().getDefiningOp<TokenGetOp>())
      if (auto rhsConst = getConstI64(subi.getRhs()))
        return buildGuard(lhs, *rhsConst, /*add=*/false);
    if (auto rhs = subi.getRhs().getDefiningOp<TokenGetOp>())
      if (auto lhsConst = getConstI64(subi.getLhs()))
        return buildGuard(rhs, *lhsConst, /*add=*/true);
  }

  if (auto addi = current.getDefiningOp<arith::AddIOp>()) {
    if (auto lhs = addi.getLhs().getDefiningOp<TokenGetOp>())
      if (auto rhsConst = getConstI64(addi.getRhs()))
        return buildGuard(lhs, *rhsConst, /*add=*/true);
    if (auto rhs = addi.getRhs().getDefiningOp<TokenGetOp>())
      if (auto lhsConst = getConstI64(addi.getLhs()))
        return buildGuard(rhs, *lhsConst, /*add=*/true);
  }

  if (auto constIdx = getConstI64(current)) {
    TokenGuard guard;
    guard.equalsValue = constIdx.value();
    guard.keyHash = hashOptionalValue(Value(), guardHashCache);
    return guard;
  }

  return std::nullopt;
}

/// Resolve all potential destinations of an emit.
static LogicalResult resolveEmitTargets(Value placeValue,
                                        SmallVectorImpl<TargetInfo> &targets) {
  if (auto ref = placeValue.getDefiningOp<PlaceRefOp>()) {
    targets.push_back(TargetInfo{ref.getPlaceAttr().getAttr(), {}});
    return success();
  }

  if (auto get = placeValue.getDefiningOp<PlaceListGetOp>()) {
    auto list = get.getList().getDefiningOp<PlaceListOp>();
    if (!list)
      return failure();
    auto placesAttr = list.getPlacesAttr();

    Value baseIndex = stripIndexCasts(get.getIndex());
    if (auto constIdx = getConstI64(baseIndex)) {
      if (*constIdx < 0 || *constIdx >= static_cast<int64_t>(placesAttr.size()))
        return failure();
      auto sym = dyn_cast<FlatSymbolRefAttr>(placesAttr[*constIdx]);
      if (!sym)
        return failure();
      targets.push_back(TargetInfo{sym.getAttr(), {}});
      return success();
    }

    for (auto [slot, attr] : llvm::enumerate(placesAttr)) {
      auto sym = dyn_cast<FlatSymbolRefAttr>(attr);
      if (!sym)
        return failure();
      TargetInfo info;
      info.symbol = sym.getAttr();
      if (auto guard = matchListIndexGuard(get.getIndex(), slot))
        if (guard->key)
          info.guards.push_back(*guard);
      targets.push_back(std::move(info));
    }
    return success();
  }

  return failure();
}

/// Follow the SSA users of `token` and collect all reachable emits.
static LogicalResult collectTokenFlows(
    Value token, SmallVectorImpl<std::pair<EmitOp, Value>> &flows,
    SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(token).second)
    return success();

  for (Operation *user : token.getUsers()) {
    if (auto emit = dyn_cast<EmitOp>(user)) {
      flows.emplace_back(emit, token);
      continue;
    }
    if (auto set = dyn_cast<TokenSetOp>(user)) {
      if (failed(collectTokenFlows(set.getResult(), flows, visited)))
        return failure();
      continue;
    }
    if (auto clone = dyn_cast<TokenCloneOp>(user)) {
      if (failed(collectTokenFlows(clone.getResult(), flows, visited)))
        return failure();
      continue;
    }
    if (isa<TokenGetOp>(user))
      continue;
    return user->emitError("unsupported token consumer in retain pass");
  }
  return success();
}

/// Recursively clone the SSA slice for `value`.
static Value cloneValueInto(Value value, IRMapping &mapping,
                            ImplicitLocOpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    arg.getOwner()->getParentOp()->emitError(
        "block arguments are unsupported in retain-observables");
    return {};
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  if (auto take = dyn_cast<TakeOp>(def)) {
    for (Value operand : take->getOperands())
      if (!cloneValueInto(operand, mapping, builder))
        return {};
    Operation *clone = builder.clone(*take, mapping);
    Value result = clone->getResult(0);
    mapping.map(value, result);
    return result;
  }

  for (Value operand : def->getOperands())
    if (!cloneValueInto(operand, mapping, builder))
      return {};

  Operation *clone = builder.clone(*def, mapping);
  auto result = cast<OpResult>(value);
  return clone->getResult(result.getResultNumber());
}

static Value buildGuardCondition(const SmallVectorImpl<TokenGuard> &guards,
                                 const EdgeTemplate &templ, Value token,
                                 ImplicitLocOpBuilder &builder) {
  if (guards.empty())
    return {};
  auto i64Ty = IntegerType::get(builder.getContext(), 64);
  IRMapping mapping;
  mapping.map(templ.takeValue, token);
  Value condition;
  for (const TokenGuard &guard : guards) {
    if (!guard.key)
      continue;
    Value keyValue = cloneValueInto(guard.key, mapping, builder);
    if (!keyValue)
      return {};
    Value lhs = builder.create<TokenGetOp>(i64Ty, token, keyValue);
    Value rhs = builder.create<arith::ConstantOp>(
        builder.getI64IntegerAttr(guard.equalsValue));
    Value eq = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, rhs);
    condition = condition ? builder.create<arith::AndIOp>(condition, eq) : eq;
  }
  return condition;
}

/// Helper to ensure a valid accumulated delay value exists.
static Value ensureDelay(Value delay, ImplicitLocOpBuilder &builder) {
  if (delay)
    return delay;
  return builder.create<arith::ConstantOp>(builder.getF64FloatAttr(0.0));
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct LPNRetainObservablesPass
    : PassWrapper<LPNRetainObservablesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNRetainObservablesPass)

  StringRef getArgument() const final { return "lpn-retain-observables"; }
  StringRef getDescription() const final {
    return "Collapse the network to only the user-marked observable places.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](NetOp net) {
      if (failed(processNet(net)))
        signalPassFailure();
    });
  }

  LogicalResult processNet(NetOp net) {
    SmallVector<PlaceOp> observablePlaces;
    DenseSet<StringAttr> observableNames;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (place.getObservableAttr()) {
        observablePlaces.push_back(place);
        observableNames.insert(place.getSymNameAttr());
      }
    if (observablePlaces.size() < 2)
      return success();

    SmallVector<std::unique_ptr<EdgeTemplate>> templates;
    DenseMap<StringAttr, SmallVector<const EdgeTemplate *>> adjacency;
    DenseMap<Value, StringAttr> takePlaces;
    for (TransitionOp trans : net.getOps<TransitionOp>()) {
      for (TakeOp take :
           llvm::make_early_inc_range(trans.getBody().getOps<TakeOp>())) {
        StringAttr source;
        if (failed(resolvePlaceSymbol(take.getPlace(), source)))
          continue;
        takePlaces[take.getResult()] = source;
        SmallVector<std::pair<EmitOp, Value>> flows;
        SmallPtrSet<Value, 8> visited;
        if (failed(collectTokenFlows(take.getResult(), flows, visited)))
          continue;
        for (auto &[emit, tokenVal] : flows) {
          SmallVector<TargetInfo> targets;
          if (failed(resolveEmitTargets(emit.getPlace(), targets)))
            continue;
          SmallVector<ControlContext> contexts;
          Operation *parent = emit.getOperation()->getParentOp();

          /// stopped when???
          while (parent && parent != trans) {
            if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
              bool inThen =
                  emit->getBlock()->getParent() == &ifOp.getThenRegion();
              contexts.push_back({ifOp.getOperation(), inThen});
            }
            parent = parent->getParentOp();
          }
          std::reverse(contexts.begin(), contexts.end());

          SmallPtrSet<Value, 8> requiredTakes;
          requiredTakes.insert(take.getResult());
          collectTakeDependencies(tokenVal, requiredTakes);
          collectTakeDependencies(emit.getPlace(), requiredTakes);
          collectTakeDependencies(emit.getDelay(), requiredTakes);
          for (const ControlContext &ctx : contexts)
            if (auto ifOp = dyn_cast<scf::IfOp>(ctx.op))
              collectTakeDependencies(ifOp.getCondition(), requiredTakes);

          bool usesHiddenTake = false;
          for (Value dep : requiredTakes) {
            auto it = takePlaces.find(dep);
            if (it == takePlaces.end())
              continue;
            if (!observableNames.contains(it->second)) {
              usesHiddenTake = true;
              break;
            }
          }
          if (usesHiddenTake)
            continue;

          // assumes all emits have the same edit???
          for (TargetInfo &target : targets) {
            SmallVector<TokenEditSignature> editSummary;
            DenseMap<Value, llvm::hash_code> hashCache;
            if (failed(summarizeTokenEdits(tokenVal, take.getResult(),
                                           editSummary, hashCache)))
              continue;
            auto templ = std::make_unique<EdgeTemplate>(
                EdgeTemplate{source, target, take.getResult(), tokenVal,
                             emit.getDelay(), contexts, editSummary});
            const EdgeTemplate *ptr = templ.get();
            adjacency[source].push_back(ptr);
            templates.push_back(std::move(templ));
          }
        }
      }
    }

    if (adjacency.empty())
      return success();

    DenseMap<StringAttr, SmallVector<EdgePath>> observablePaths;
    for (PlaceOp place : observablePlaces) {
      StringAttr root = place.getSymNameAttr();
      DenseSet<StringAttr> visited;
      visited.insert(root);
      EdgePath prefix;
      dfsPaths(root, root, visited, prefix, observableNames, adjacency,
               observablePaths);
    }

    dedupPaths(observablePaths);

    if (observablePaths.empty())
      return success();

    HaltOp halt = nullptr;
    for (Operation &op : net.getBody().getOps())
      if ((halt = dyn_cast<HaltOp>(&op)))
        break;
    if (!halt)
      return net.emitError("net missing lpn.halt terminator");

    SmallVector<Operation *> originalTransitions;
    for (TransitionOp trans : net.getOps<TransitionOp>())
      originalTransitions.push_back(trans);
    SmallVector<Operation *> removablePlaces;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (!place.getObservableAttr())
        removablePlaces.push_back(place);

    Block &body = net.getBody().front();
    OpBuilder topBuilder(&body, Block::iterator(halt));
    MLIRContext *ctx = net.getContext();
    for (auto &entry : observablePaths) {
      if (entry.second.empty())
        continue;
      std::string transName = (entry.first.getValue() + "_retain").str();
      auto trans =
          topBuilder.create<TransitionOp>(net.getLoc(),
                                          topBuilder.getStringAttr(transName));
      Region &region = trans.getBody();
      auto *block = new Block();
      region.push_back(block);
      ImplicitLocOpBuilder builder(net.getLoc(), ctx);
      builder.setInsertionPointToStart(block);
      auto placeRef = builder.create<PlaceRefOp>(
          PlaceType::get(ctx), FlatSymbolRefAttr::get(entry.first));
      Value seed = builder.create<TakeOp>(TokenType::get(ctx), placeRef);
      SmallVector<PathCursor, 8> cursors;
      for (const EdgePath &path : entry.second)
        cursors.push_back(PathCursor{&path, 0, 0});
      if (failed(emitCursorSet(cursors, seed, Value(), builder)))
        return failure();
      builder.create<ScheduleReturnOp>();
    }

    for (Operation *op : originalTransitions)
      op->erase();
    for (Operation *op : removablePlaces)
      op->erase();

    simplifyChoiceLadders(net);
    return success();
  }

  const EdgeTemplate *getTemplate(const PathCursor &cursor) const {
    if (!cursor.path || cursor.edgeIndex >= cursor.path->size())
      return nullptr;
    return (*(cursor.path))[cursor.edgeIndex];
  }

  void dfsPaths(
      StringAttr root, StringAttr current, DenseSet<StringAttr> &visited,
      EdgePath &prefix, const DenseSet<StringAttr> &observables,
      DenseMap<StringAttr, SmallVector<const EdgeTemplate *>> &adjacency,
      DenseMap<StringAttr, SmallVector<EdgePath>> &paths) const {
    if (observables.contains(current) && current != root) {
      paths[root].push_back(prefix);
      return;
    }
    auto it = adjacency.find(current);
    if (it == adjacency.end())
      return;
    for (const EdgeTemplate *templ : it->second) {
      StringAttr next = templ->target.symbol;
      if (!visited.insert(next).second)
        continue;
      prefix.push_back(templ);
      dfsPaths(root, next, visited, prefix, observables, adjacency, paths);
      prefix.pop_back();
      visited.erase(next);
    }
  }

  struct ContextGroup {
    const EdgeTemplate *representative = nullptr;
    SmallVector<PathCursor, 4> thenCursors;
    SmallVector<PathCursor, 4> elseCursors;
  };

  LogicalResult emitCursorSet(ArrayRef<PathCursor> cursors, Value token,
                              Value accumulatedDelay,
                              ImplicitLocOpBuilder &builder) const {
    if (cursors.empty())
      return success();

    DenseMap<Operation *, ContextGroup> contextGroups;
    SmallVector<PathCursor, 8> ready;
    for (const PathCursor &cursor : cursors) {
      const EdgeTemplate *templ = getTemplate(cursor);
      if (!templ)
        continue;
      if (cursor.contextIndex >= templ->contexts.size()) {
        ready.push_back(cursor);
        continue;
      }
      const ControlContext &ctx = templ->contexts[cursor.contextIndex];
      auto &group = contextGroups[ctx.op];
      if (!group.representative)
        group.representative = templ;
      PathCursor next = cursor;
      next.contextIndex++;
      if (ctx.isThen)
        group.thenCursors.push_back(next);
      else
        group.elseCursors.push_back(next);
    }

    for (auto &entry : contextGroups) {
      Operation *ctxOp = entry.first;
      if (entry.second.thenCursors.empty() ||
          entry.second.elseCursors.empty()) {
        ArrayRef<PathCursor> active =
            entry.second.thenCursors.empty() ? entry.second.elseCursors
                                             : entry.second.thenCursors;
        if (!active.empty())
          if (failed(emitCursorSet(active, token, accumulatedDelay, builder)))
            return failure();
        continue;
      }
      if (ctxOp->hasAttr("lpn.hidden_choice")) {
        if (failed(emitChoice(entry.second, token, accumulatedDelay, builder)))
          return failure();
        continue;
      }
      auto ifOp = dyn_cast<scf::IfOp>(ctxOp);
      if (!ifOp)
        return builder.getInsertionBlock()->getParentOp()->emitError(
            "unsupported control context");
      const EdgeTemplate *repr = entry.second.representative;
      IRMapping mapping;
      mapping.map(repr->takeValue, token);
      Value cond = cloneValueInto(ifOp.getCondition(), mapping, builder);
      if (!cond)
        return failure();
      bool hasElse = !entry.second.elseCursors.empty();
      scf::IfOp clonedIf =
          builder.create<scf::IfOp>(TypeRange(), cond, hasElse);
      auto &thenBlock = clonedIf.getThenRegion().front();
      thenBlock.clear();
      ImplicitLocOpBuilder thenBuilder(builder.getLoc(), builder.getContext());
      thenBuilder.setInsertionPointToStart(&thenBlock);
      if (failed(emitCursorSet(entry.second.thenCursors, token,
                               accumulatedDelay, thenBuilder)))
        return failure();
      thenBuilder.create<scf::YieldOp>();
      if (hasElse) {
        auto &elseBlock = clonedIf.getElseRegion().front();
        elseBlock.clear();
        ImplicitLocOpBuilder elseBuilder(builder.getLoc(),
                                         builder.getContext());
        elseBuilder.setInsertionPointToStart(&elseBlock);
        if (failed(emitCursorSet(entry.second.elseCursors, token,
                                 accumulatedDelay, elseBuilder)))
          return failure();
        elseBuilder.create<scf::YieldOp>();
      }
      builder.setInsertionPointAfter(clonedIf);
    }

    SmallVector<PathCursor, 8> dedupReady;
    SmallVector<const EdgeTemplate *, 4> seenLeaves;
    for (const PathCursor &cursor : ready) {
      const EdgeTemplate *templ = getTemplate(cursor);
      if (!templ) {
        dedupReady.push_back(cursor);
        continue;
      }
      bool last = cursor.path && (cursor.edgeIndex + 1 == cursor.path->size());
      if (!last) {
        dedupReady.push_back(cursor);
        continue;
      }
      bool duplicate = false;
      for (const EdgeTemplate *seen : seenLeaves)
        if (equivalentTemplate(templ, seen)) {
          duplicate = true;
          break;
        }
      if (duplicate)
        continue;
      seenLeaves.push_back(templ);
      dedupReady.push_back(cursor);
    }

    return emitLeafChoices(dedupReady, token, accumulatedDelay, builder);
  }

  LogicalResult emitLeaf(const PathCursor &cursor, Value token,
                         Value accumulatedDelay,
                         ImplicitLocOpBuilder &builder) const {
    const EdgeTemplate *templ = getTemplate(cursor);
    if (!templ)
      return success();

    auto emitBody = [&](ImplicitLocOpBuilder &inner, Value currentToken,
                        Value currentDelay) -> LogicalResult {
      IRMapping mapping;
      mapping.map(templ->takeValue, currentToken);
      Value newToken = cloneValueInto(templ->tokenValue, mapping, inner);
      if (!newToken)
        return failure();
      Value stepDelay = cloneValueInto(templ->delayValue, mapping, inner);
      if (!stepDelay)
        return failure();
      Value totalDelay = ensureDelay(currentDelay, inner);
      totalDelay =
          inner.create<arith::AddFOp>(totalDelay, stepDelay).getResult();
      bool last = (cursor.edgeIndex + 1 == cursor.path->size());
      if (last) {
        auto placeType = PlaceType::get(inner.getContext());
        auto placeAttr = FlatSymbolRefAttr::get(templ->target.symbol);
        Value place = inner.create<PlaceRefOp>(placeType, placeAttr);
        inner.create<EmitOp>(place, newToken, totalDelay);
        return success();
      }
      PathCursor next{cursor.path, cursor.edgeIndex + 1, 0};
      SmallVector<PathCursor, 1> nextSet{next};
      return emitCursorSet(nextSet, newToken, totalDelay, inner);
    };

    if (templ->target.guards.empty())
      return emitBody(builder, token, accumulatedDelay);

    Value cond =
        buildGuardCondition(templ->target.guards, *templ, token, builder);
    if (!cond)
      return builder.getInsertionBlock()->getParentOp()->emitError(
          "failed to build guard condition");
    scf::IfOp guardIf =
        builder.create<scf::IfOp>(TypeRange(), cond, /*withElse=*/false);
    auto &guardBlock = guardIf.getThenRegion().front();
    guardBlock.clear();
    ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
    inner.setInsertionPointToStart(&guardBlock);
    if (failed(emitBody(inner, token, accumulatedDelay)))
      return failure();
    inner.create<scf::YieldOp>();
    builder.setInsertionPointAfter(guardIf);
    return success();
  }

  LogicalResult emitLeafChoices(ArrayRef<PathCursor> cursors, Value token,
                                Value accumulatedDelay,
                                ImplicitLocOpBuilder &builder) const {
    if (cursors.empty())
      return success();
    if (cursors.size() == 1)
      return emitLeaf(cursors.front(), token, accumulatedDelay, builder);
    auto choice = builder.create<ChoiceOp>();
    auto populate = [&](Region &region,
                        ArrayRef<PathCursor> subset) -> LogicalResult {
      region.getBlocks().clear();
      Block &block = region.emplaceBlock();
      ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                         builder.getContext());
      branchBuilder.setInsertionPointToStart(&block);
      if (failed(
              emitLeafChoices(subset, token, accumulatedDelay, branchBuilder)))
        return failure();
      branchBuilder.create<ChoiceYieldOp>();
      return success();
    };
    if (failed(populate(choice.getThenRegion(), cursors.take_front(1))))
      return failure();
    if (failed(populate(choice.getElseRegion(), cursors.drop_front())))
      return failure();
    builder.setInsertionPointAfter(choice);
    return success();
  }

  LogicalResult emitChoice(const ContextGroup &group, Value token,
                           Value accumulatedDelay,
                           ImplicitLocOpBuilder &builder) const {
    auto choice = builder.create<ChoiceOp>();
    auto populate = [&](Region &region,
                        ArrayRef<PathCursor> cursors) -> LogicalResult {
      region.getBlocks().clear();
      Block &block = region.emplaceBlock();
      ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                         builder.getContext());
      branchBuilder.setInsertionPointToStart(&block);
      if (!cursors.empty())
        if (failed(
                emitCursorSet(cursors, token, accumulatedDelay, branchBuilder)))
          return failure();
      branchBuilder.create<ChoiceYieldOp>();
      return success();
    };

    if (failed(populate(choice.getThenRegion(), group.thenCursors)))
      return failure();
    if (failed(populate(choice.getElseRegion(), group.elseCursors)))
      return failure();
    builder.setInsertionPointAfter(choice);
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNRetainObservablesPass() {
  return std::make_unique<LPNRetainObservablesPass>();
}

} // namespace mlir::lpn
