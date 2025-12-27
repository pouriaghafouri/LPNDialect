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
#include <cassert>

namespace mlir::lpn {
namespace {

//===----------------------------------------------------------------------===//
// Pass overview
//===----------------------------------------------------------------------===//
//
// The hypergraph retain pass collapses an arbitrary LPN network down to the
// user-marked observable places without making structural assumptions such as
// “each place has at most one producer and consumer”.  Instead, it treats the
// network as a hypergraph whose vertices are observable places and whose
// hyperedges capture the exact set of observable tokens consumed together:
//
//   1. For every `lpn.take` result we enumerate all reachable `lpn.emit`
//      operations by walking the SSA use-def chain.  The SSA slice from the
//      take to each emit is summarized as an EdgeTemplate that records the
//      canonical driver observable, the multiset of observable takes used,
//      the cloned token/delay expressions, and metadata (guards, edit log,
//      control context, and expression fingerprints).
//
//   2. We cluster identical EdgeTemplates per driver place so structurally
//      equivalent hyperedges are stored only once.  Equivalence requires the
//      same driver, observable source set, guard predicates, edit signatures,
//      SSA fingerprints, and nesting of control operations.
//
//   3. The hyperedges induce a directed graph between observables.  We walk
//      this graph via DFS to enumerate every observable-to-observable path.
//      Paths that revisit an observable terminate, ensuring cycles do not
//      explode the search space.
//
//   4. For each observable root with outgoing paths we synthesize a retained
//      transition.  The transition takes from the root once, then replays each
//      path by rematerializing the recorded slices.  Additional observables
//      required by a hyperedge are taken exactly once per synthesized firing;
//      guards become `scf.if`, arbitrary control constructs (e.g., nested ifs)
//      are cloned, and per-edge delays are accumulated.
//
// Because hyperedges track multi-source consumption explicitly, the pass can
// retain nets where transitions require multiple observable tokens at once,
// where emissions are routed via data-dependent control flow, and where
// different latent paths share large sub-slices.  The resulting retained net
// is minimal with respect to the chosen observables yet faithful to both the
// token edits and the causal relationships encoded in the original MLIR.
//
//   * Before enumerating observable paths we cluster identical hyperedges per
//     driver place.  Hyperedges with the same driver, observable source set,
//     control contexts, token edits, guard predicates, and delay/token SSA slices
//     collapse to a single template.  This ensures the retained network does not
//     duplicate transitions just because several paths in the original net were
//     structurally identical.

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

static bool blockInRegion(Block *block, Region &region) {
  for (Region *current = block ? block->getParent() : nullptr; current;
       current = current->getParentRegion())
    if (current == &region)
      return true;
  return false;
}

/// Single equality predicate derived from token metadata.
struct TokenGuard {
  StringAttr field;
  int64_t equalsValue;
};

/// Destination place plus optional guards describing the routing condition.
struct TargetInfo {
  StringAttr symbol;
  SmallVector<TokenGuard> guards;
};

struct TokenEditSignature {
  StringAttr field;
  llvm::hash_code valueHash;
  SmallVector<unsigned, 4> sourceRefs;
};

enum class ContextKind { IfOp, ChoiceOp, ForOp };

struct ControlContext {
  Operation *op;
  ContextKind kind;
  bool isThen;
};

struct ObservableSource {
  StringAttr place;
  Value takeValue;
};

/// SSA stencil for a single take/emit pair.
struct EdgeTemplate {
  StringAttr driver;
  Value driverTake;
  TargetInfo target;
  SmallVector<ObservableSource> sources;
  Value tokenValue;
  Value delayValue;
  SmallVector<ControlContext> contexts;
  SmallVector<TokenEditSignature> editSummary;
  llvm::hash_code tokenHash;
  llvm::hash_code delayHash;
};

/// Observable-to-observable chain of edges.
using EdgePath = SmallVector<const EdgeTemplate *>;

struct PathCursor {
  const EdgePath *path;
  size_t edgeIndex;
  size_t contextIndex;
};

static bool guardsEqual(ArrayRef<TokenGuard> lhs, ArrayRef<TokenGuard> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (a.field != b.field || a.equalsValue != b.equalsValue)
      return false;
  return true;
}

static bool editsEqual(ArrayRef<TokenEditSignature> lhs,
                       ArrayRef<TokenEditSignature> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (a.field != b.field || a.valueHash != b.valueHash)
      return false;
  return true;
}

static bool contextsEqual(ArrayRef<ControlContext> lhs,
                          ArrayRef<ControlContext> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (a.op != b.op || a.isThen != b.isThen)
      return false;
  return true;
}

static bool equivalentTemplate(const EdgeTemplate *lhs,
                               const EdgeTemplate *rhs) {
  if (lhs == rhs)
    return true;
  if (lhs->driver != rhs->driver)
    return false;
  if (lhs->sources.size() != rhs->sources.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs->sources, rhs->sources))
    if (a.place != b.place)
      return false;
  if (lhs->target.symbol != rhs->target.symbol)
    return false;
  if (!guardsEqual(lhs->target.guards, rhs->target.guards))
    return false;
  if (!editsEqual(lhs->editSummary, rhs->editSummary))
    return false;
  if (!contextsEqual(lhs->contexts, rhs->contexts))
    return false;
  if (lhs->tokenHash != rhs->tokenHash || lhs->delayHash != rhs->delayHash)
    return false;
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

static void clusterHyperedges(
    DenseMap<StringAttr, SmallVector<const EdgeTemplate *>> &adjacency) {
  for (auto &entry : adjacency) {
    SmallVector<const EdgeTemplate *> unique;
    for (const EdgeTemplate *templ : entry.second) {
      bool duplicate = false;
      for (const EdgeTemplate *seen : unique)
        if (equivalentTemplate(templ, seen)) {
          duplicate = true;
          break;
        }
      if (!duplicate)
        unique.push_back(templ);
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
    DenseMap<Value, llvm::hash_code> &hashCache,
    const SmallVector<ObservableSource> &sources) {
  if (current == source)
    return success();
  Operation *def = current.getDefiningOp();
  if (!def)
    return failure();
  if (auto set = dyn_cast<TokenSetOp>(def)) {
    if (failed(summarizeTokenEdits(set.getToken(), source, edits, hashCache,
                                   sources)))
      return failure();
    TokenEditSignature sig;
    sig.field = set.getFieldAttr();
    sig.valueHash = hashValueExpr(set.getValue(), hashCache);
    auto recordRefs = [&](auto &&self, Value v) -> void {
      if (!v)
        return;
      if (auto get = v.getDefiningOp<TokenGetOp>()) {
        for (auto [idx, src] : llvm::enumerate(sources))
          if (src.takeValue == get.getToken())
            sig.sourceRefs.push_back(idx);
      }
      if (Operation *producer = v.getDefiningOp())
        for (Value operand : producer->getOperands())
          self(self, operand);
    };
    recordRefs(recordRefs, set.getValue());
    edits.push_back(std::move(sig));
    return success();
  }
  if (auto clone = dyn_cast<TokenCloneOp>(def))
    return summarizeTokenEdits(clone.getToken(), source, edits, hashCache,
                               sources);
  return def->emitError(
      "token flow includes unsupported op while summarizing");
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
  Value current = stripIndexCasts(index);

  if (auto get = current.getDefiningOp<TokenGetOp>())
    return TokenGuard{get.getFieldAttr(), slot};

  auto buildGuard = [&](TokenGetOp get, int64_t offset,
                        bool add) -> std::optional<TokenGuard> {
    int64_t base = add ? slot - offset : slot + offset;
    return TokenGuard{get.getFieldAttr(), base};
  };

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

  if (auto constIdx = getConstI64(current))
    return TokenGuard{StringAttr{}, constIdx.value()};

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
        if (guard->field)
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
    if (Value mapped = mapping.lookupOrNull(arg))
      return mapped;
    arg.getOwner()->getParentOp()->emitError(
        "unmapped block argument while cloning hypergraph slice");
    return {};
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  if (auto take = dyn_cast<TakeOp>(def)) {
    if (Value mapped = mapping.lookupOrNull(take.getResult()))
      return mapped;
    take.emitError("unmapped lpn.take while cloning hypergraph slice");
    return {};
  }

  for (Value operand : def->getOperands())
    if (!cloneValueInto(operand, mapping, builder))
      return {};

  Operation *clone = builder.clone(*def, mapping);
  auto result = cast<OpResult>(value);
  return clone->getResult(result.getResultNumber());
}

static Value buildGuardCondition(const SmallVectorImpl<TokenGuard> &guards,
                                 Value token, ImplicitLocOpBuilder &builder) {
  if (guards.empty())
    return {};
  auto i64Ty = IntegerType::get(builder.getContext(), 64);
  Value condition;
  for (const TokenGuard &guard : guards) {
    if (!guard.field)
      continue;
    Value lhs = builder.create<TokenGetOp>(i64Ty, token, guard.field);
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

using TokenEnv = DenseMap<StringAttr, SmallVector<Value>>;
using TakeEnv = DenseMap<Value, Value>;
using SSAEnv = DenseMap<Value, Value>;

static void mapContextValues(const SSAEnv &env, IRMapping &mapping) {
  for (const auto &entry : env)
    mapping.map(entry.first, entry.second);
}

static void mapTemplateSources(const EdgeTemplate *templ, IRMapping &mapping,
                               const TakeEnv &takes) {
  for (const ObservableSource &src : templ->sources)
    if (Value mapped = takes.lookup(src.takeValue))
      mapping.map(src.takeValue, mapped);
}

static LogicalResult ensureTemplateSources(const EdgeTemplate *templ,
                                           Value driverToken, TokenEnv &tokens,
                                           TakeEnv &takes,
                                           ImplicitLocOpBuilder &builder) {
  auto placeType = PlaceType::get(builder.getContext());
  auto tokenType = TokenType::get(builder.getContext());
  for (const ObservableSource &src : templ->sources) {
    if (takes.contains(src.takeValue))
      continue;
    if (src.takeValue == templ->driverTake) {
      takes[src.takeValue] = driverToken;
      continue;
    }
    SmallVector<Value> &queue = tokens[src.place];
    Value token;
    if (!queue.empty()) {
      token = queue.pop_back_val();
    } else {
      Value ref = builder.create<PlaceRefOp>(placeType,
                                             FlatSymbolRefAttr::get(src.place));
      token = builder.create<TakeOp>(tokenType, ref);
    }
    takes[src.takeValue] = token;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct LPNRetainHypergraphPass
    : PassWrapper<LPNRetainHypergraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNRetainHypergraphPass)

  StringRef getArgument() const final { return "lpn-retain-hypergraph"; }
  StringRef getDescription() const final {
    return "EXPERIMENTAL: hypergraph-based observable reduction.";
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
    DenseSet<Value> hiddenTakeResults;
    for (TransitionOp trans : net.getOps<TransitionOp>()) {
      SmallVector<TakeOp, 8> takes;
      for (TakeOp take : trans.getBody().getOps<TakeOp>())
        takes.push_back(take);
      for (TakeOp take : takes) {
        StringAttr place;
        if (failed(resolvePlaceSymbol(take.getPlace(), place)))
          continue;
        if (observableNames.contains(place))
          takePlaces[take.getResult()] = place;
        else
          hiddenTakeResults.insert(take.getResult());
      }
      for (TakeOp take : takes) {
        StringAttr source = takePlaces.lookup(take.getResult());
        if (!source)
          continue;
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

          while (parent && parent != trans) {
            Block *emitBlock = emit->getBlock();
            if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
              Region &thenRegion = ifOp.getThenRegion();
              Region &elseRegion = ifOp.getElseRegion();
              bool inThen = blockInRegion(emitBlock, thenRegion);
              bool inElse = blockInRegion(emitBlock, elseRegion);
              assert((inThen || inElse) &&
                     "emit block must belong to either then or else region");
              bool hidden = ifOp->hasAttr("lpn.hidden_choice");
              contexts.push_back({ifOp.getOperation(),
                                  hidden ? ContextKind::ChoiceOp
                                         : ContextKind::IfOp,
                                  inThen});
            } else if (auto choice = dyn_cast<ChoiceOp>(parent)) {
              Region &thenRegion = choice.getThenRegion();
              Region &elseRegion = choice.getElseRegion();
              bool inThen = blockInRegion(emitBlock, thenRegion);
              bool inElse = blockInRegion(emitBlock, elseRegion);
              assert((inThen || inElse) &&
                     "emit block must belong to either choice branch");
              contexts.push_back(
                  {choice.getOperation(), ContextKind::ChoiceOp, inThen});
            } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
              Region &body = forOp.getRegion();
              assert(blockInRegion(emitBlock, body) &&
                     "loop body must contain emit");
              contexts.push_back(
                  {forOp.getOperation(), ContextKind::ForOp, true});
            }
            parent = parent->getParentOp();
          }
          std::reverse(contexts.begin(), contexts.end());

          SmallPtrSet<Value, 8> requiredTakes;
          requiredTakes.insert(take.getResult());
          collectTakeDependencies(tokenVal, requiredTakes);
          collectTakeDependencies(emit.getPlace(), requiredTakes);
          collectTakeDependencies(emit.getDelay(), requiredTakes);
          for (const ControlContext &ctx : contexts) {
            if (ctx.kind == ContextKind::IfOp) {
              if (auto ifOp = dyn_cast<scf::IfOp>(ctx.op))
                collectTakeDependencies(ifOp.getCondition(), requiredTakes);
              continue;
            }
            if (ctx.kind == ContextKind::ForOp) {
              auto forOp = dyn_cast<scf::ForOp>(ctx.op);
              collectTakeDependencies(forOp.getLowerBound(), requiredTakes);
              collectTakeDependencies(forOp.getUpperBound(), requiredTakes);
              collectTakeDependencies(forOp.getStep(), requiredTakes);
            }
          }

          SmallVector<ObservableSource> sources;
          for (Value dep : requiredTakes) {
            auto it = takePlaces.find(dep);
            if (it == takePlaces.end())
              continue;
            sources.push_back(ObservableSource{it->second, dep});
          }
          if (sources.empty())
            continue;
          llvm::sort(sources, [](const ObservableSource &a,
                                 const ObservableSource &b) {
            if (a.place != b.place)
              return a.place.getValue() < b.place.getValue();
            return a.takeValue.getAsOpaquePointer() <
                   b.takeValue.getAsOpaquePointer();
          });
          StringAttr driver = sources.front().place;
          if (driver != source)
            continue;

          for (TargetInfo &target : targets) {
            SmallVector<ObservableSource> sourcesCopy = sources;
            SmallVector<TokenEditSignature> editSummary;
            DenseMap<Value, llvm::hash_code> hashCache;
            if (failed(summarizeTokenEdits(tokenVal, take.getResult(),
                                           editSummary, hashCache, sources)))
              continue;
            llvm::hash_code tokenHash = hashOptionalValue(tokenVal, hashCache);
            llvm::hash_code delayHash =
                hashOptionalValue(emit.getDelay(), hashCache);
            auto templ = std::make_unique<EdgeTemplate>(
                EdgeTemplate{driver,
                             take.getResult(),
                             target,
                             std::move(sourcesCopy),
                             tokenVal,
                             emit.getDelay(),
                             contexts,
                             editSummary,
                             tokenHash,
                             delayHash});
            const EdgeTemplate *ptr = templ.get();
            adjacency[driver].push_back(ptr);
            templates.push_back(std::move(templ));
          }
        }
      }
    }

    clusterHyperedges(adjacency);

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
      TokenEnv tokens;
      tokens[entry.first] = SmallVector<Value>{seed};
      TakeEnv takes;
      for (const PathCursor &cursor : cursors) {
        TokenEnv pathTokens = tokens;
        TakeEnv pathTakes = takes;
        SSAEnv pathSSA;
        if (failed(processCursor(cursor, seed, Value(), std::move(pathTokens),
                                 std::move(pathTakes), std::move(pathSSA),
                                 builder)))
          return failure();
      }
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

LogicalResult processCursor(PathCursor cursor, Value token,
                            Value accumulatedDelay, TokenEnv tokens,
                            TakeEnv takes, SSAEnv ssaEnv,
                            ImplicitLocOpBuilder &builder) const {
  const EdgeTemplate *templ = getTemplate(cursor);
  if (!templ)
    return success();
  if (cursor.contextIndex < templ->contexts.size()) {
    const ControlContext &ctx = templ->contexts[cursor.contextIndex];
    if (ctx.kind == ContextKind::ChoiceOp)
      return emitChoiceBranch(cursor, templ, ctx.isThen, token,
                              accumulatedDelay, std::move(tokens),
                              std::move(takes), std::move(ssaEnv), builder);
    if (ctx.kind == ContextKind::ForOp) {
      auto forOp = dyn_cast<scf::ForOp>(ctx.op);
      if (!forOp)
        return builder.getInsertionBlock()->getParentOp()->emitError(
            "malformed loop context");
      return emitForBranch(cursor, templ, forOp, token, accumulatedDelay,
                           std::move(tokens), std::move(takes),
                           std::move(ssaEnv), builder);
    }
    auto ifOp = dyn_cast<scf::IfOp>(ctx.op);
    if (!ifOp)
      return builder.getInsertionBlock()->getParentOp()->emitError(
          "unsupported control context");
    return emitIfBranch(cursor, templ, ifOp, ctx.isThen, token,
                        accumulatedDelay, std::move(tokens), std::move(takes),
                        std::move(ssaEnv), builder);
  }
  return emitLeaf(cursor, templ, token, accumulatedDelay, std::move(tokens),
                  std::move(takes), std::move(ssaEnv), builder);
}

LogicalResult emitIfBranch(PathCursor cursor, const EdgeTemplate *templ,
                           scf::IfOp ifOp, bool takeThen, Value token,
                           Value accumulatedDelay, TokenEnv tokens,
                           TakeEnv takes, SSAEnv ssaEnv,
                           ImplicitLocOpBuilder &builder) const {
  TokenEnv condTokens = tokens;
  TakeEnv condTakes = takes;
  if (failed(
          ensureTemplateSources(templ, token, condTokens, condTakes, builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(ssaEnv, mapping);
  mapTemplateSources(templ, mapping, condTakes);
  Value cond = cloneValueInto(ifOp.getCondition(), mapping, builder);
  if (!cond)
    return failure();
  scf::IfOp clonedIf =
      builder.create<scf::IfOp>(TypeRange(), cond, /*withElse=*/true);
  auto populate = [&](Region &region, bool active, TokenEnv branchTokens,
                      TakeEnv branchTakes, SSAEnv branchEnv) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                       builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (active) {
      PathCursor next = cursor;
      next.contextIndex++;
      if (failed(processCursor(next, token, accumulatedDelay,
                               std::move(branchTokens),
                               std::move(branchTakes), std::move(branchEnv),
                               branchBuilder)))
        return failure();
    }
    branchBuilder.create<scf::YieldOp>();
    return success();
  };
  if (failed(populate(clonedIf.getThenRegion(), takeThen, condTokens,
                      condTakes, ssaEnv)))
    return failure();
  if (failed(populate(clonedIf.getElseRegion(), !takeThen, std::move(tokens),
                      std::move(takes), std::move(ssaEnv))))
    return failure();
  builder.setInsertionPointAfter(clonedIf);
  return success();
}

LogicalResult emitChoiceBranch(PathCursor cursor, const EdgeTemplate *templ,
                               bool useThen, Value token,
                               Value accumulatedDelay, TokenEnv tokens,
                               TakeEnv takes, SSAEnv ssaEnv,
                               ImplicitLocOpBuilder &builder) const {
  auto choice = builder.create<ChoiceOp>();
  auto populate = [&](Region &region, bool active, TokenEnv branchTokens,
                      TakeEnv branchTakes, SSAEnv branchEnv) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                       builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (active) {
      PathCursor next = cursor;
      next.contextIndex++;
      if (failed(processCursor(next, token, accumulatedDelay,
                               std::move(branchTokens),
                               std::move(branchTakes), std::move(branchEnv),
                               branchBuilder)))
        return failure();
    }
    branchBuilder.create<ChoiceYieldOp>();
    return success();
  };
  if (failed(populate(choice.getThenRegion(), useThen, tokens, takes, ssaEnv)))
    return failure();
  if (failed(populate(choice.getElseRegion(), !useThen, std::move(tokens),
                      std::move(takes), std::move(ssaEnv))))
    return failure();
  builder.setInsertionPointAfter(choice);
  return success();
}

LogicalResult emitForBranch(PathCursor cursor, const EdgeTemplate *templ,
                            scf::ForOp forOp, Value token,
                            Value accumulatedDelay, TokenEnv tokens,
                            TakeEnv takes, SSAEnv ssaEnv,
                            ImplicitLocOpBuilder &builder) const {
  if (forOp.getNumRegionIterArgs() != 0)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "retain pass does not support scf.for with iter_args");
  TokenEnv loopTokens = tokens;
  TakeEnv loopTakes = takes;
  if (failed(ensureTemplateSources(templ, token, loopTokens, loopTakes,
                                   builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(ssaEnv, mapping);
  mapTemplateSources(templ, mapping, loopTakes);
  Value lower = cloneValueInto(forOp.getLowerBound(), mapping, builder);
  Value upper = cloneValueInto(forOp.getUpperBound(), mapping, builder);
  Value step = cloneValueInto(forOp.getStep(), mapping, builder);
  if (!lower || !upper || !step)
    return failure();
  scf::ForOp cloned = builder.create<scf::ForOp>(lower, upper, step);
  Block &body = cloned.getRegion().front();
  body.clear();
  ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
  inner.setInsertionPointToStart(&body);
  SSAEnv innerEnv = ssaEnv;
  innerEnv[forOp.getBody()->getArgument(0)] = cloned.getInductionVar();
  PathCursor next = cursor;
  next.contextIndex++;
  if (failed(processCursor(next, token, accumulatedDelay,
                           std::move(loopTokens), std::move(loopTakes),
                           std::move(innerEnv), inner)))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult emitLeaf(PathCursor cursor, const EdgeTemplate *templ,
                       Value token, Value accumulatedDelay, TokenEnv tokens,
                       TakeEnv takes, SSAEnv ssaEnv,
                       ImplicitLocOpBuilder &builder) const {
  auto emitBody = [&](ImplicitLocOpBuilder &inner, Value currentToken,
                      Value currentDelay, TokenEnv innerTokens,
                      TakeEnv innerTakes, SSAEnv innerEnv) -> LogicalResult {
    if (failed(ensureTemplateSources(templ, currentToken, innerTokens,
                                     innerTakes, inner)))
      return failure();
    IRMapping mapping;
    mapContextValues(innerEnv, mapping);
    mapTemplateSources(templ, mapping, innerTakes);
    Value newToken = cloneValueInto(templ->tokenValue, mapping, inner);
    if (!newToken)
      return failure();
    Value stepDelay = cloneValueInto(templ->delayValue, mapping, inner);
    if (!stepDelay)
      return failure();
    Value totalDelay = ensureDelay(currentDelay, inner);
    totalDelay =
        inner.create<arith::AddFOp>(totalDelay, stepDelay).getResult();
    innerTokens[templ->target.symbol].push_back(newToken);
    bool last = cursor.path && (cursor.edgeIndex + 1 == cursor.path->size());
    if (last) {
      auto placeType = PlaceType::get(inner.getContext());
      auto placeAttr = FlatSymbolRefAttr::get(templ->target.symbol);
      Value place = inner.create<PlaceRefOp>(placeType, placeAttr);
      inner.create<EmitOp>(place, newToken, totalDelay);
      return success();
    }
    PathCursor next{cursor.path, cursor.edgeIndex + 1, 0};
    return processCursor(next, newToken, totalDelay, std::move(innerTokens),
                         std::move(innerTakes), std::move(innerEnv), inner);
  };

  Value cond = buildGuardCondition(templ->target.guards, token, builder);
  if (!cond)
    return emitBody(builder, token, accumulatedDelay, std::move(tokens),
                    std::move(takes), std::move(ssaEnv));

  scf::IfOp guardIf =
      builder.create<scf::IfOp>(TypeRange(), cond, /*withElse=*/false);
  auto &guardBlock = guardIf.getThenRegion().front();
  guardBlock.clear();
  ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
  inner.setInsertionPointToStart(&guardBlock);
  if (failed(emitBody(inner, token, accumulatedDelay, std::move(tokens),
                      std::move(takes), std::move(ssaEnv))))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(guardIf);
  return success();
}
};

} // namespace

std::unique_ptr<Pass> createLPNRetainHypergraphPass() {
  return std::make_unique<LPNRetainHypergraphPass>();
}

} // namespace mlir::lpn
