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

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

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
    ArrayRef<ObservableSource> sources) {
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

struct CursorState {
  PathCursor cursor;
  Value token;
  Value delay;
  TokenEnv tokens;
  TakeEnv takes;
  SSAEnv ssa;
};

struct ContextGroup {
  Operation *op = nullptr;
  ContextKind kind = ContextKind::IfOp;
  SmallVector<CursorState, 4> thenStates;
  SmallVector<CursorState, 4> elseStates;
  SmallVector<CursorState, 4> bodyStates;
};

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
    DenseMap<Value, unsigned> takeGuardIds;
    DenseMap<unsigned, ObservableSource> guardIdSources;
    for (TransitionOp trans : net.getOps<TransitionOp>()) {
      SmallVector<TakeOp, 8> takes;
      for (TakeOp take : trans.getBody().getOps<TakeOp>())
        takes.push_back(take);
      SmallVector<ObservableSource> transitionSources;
      for (TakeOp take : takes) {
        StringAttr place;
        if (failed(resolvePlaceSymbol(take.getPlace(), place)))
          continue;
        takePlaces[take.getResult()] = place;
        transitionSources.push_back({place, take.getResult()});
        if (auto guardAttr =
                take->getAttrOfType<IntegerAttr>(kGuardIdAttr)) {
          unsigned guardId = guardAttr.getInt();
          takeGuardIds[take.getResult()] = guardId;
          guardIdSources[guardId] = {place, take.getResult()};
        }
      }
      for (TakeOp take : takes) {
        StringAttr driverPlace = takePlaces.lookup(take.getResult());
        if (!driverPlace)
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

          SmallVector<SmallVector<ObservableSource, 4>, 4> candidateSources;
          bool usedGuardMetadata = false;
          if (auto guardAttr =
                  emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr)) {
            auto guardIt = takeGuardIds.find(take.getResult());
            if (guardIt != takeGuardIds.end()) {
              unsigned driverGuardId = guardIt->second;
              for (Attribute attr : guardAttr) {
                auto pathAttr = dyn_cast<ArrayAttr>(attr);
                if (!pathAttr)
                  continue;
                SmallVector<unsigned, 4> guardIds;
                guardIds.reserve(pathAttr.size());
                bool containsDriver = false;
                for (Attribute elem : pathAttr) {
                  auto intAttr = dyn_cast<IntegerAttr>(elem);
                  if (!intAttr)
                    continue;
                  unsigned guardId = intAttr.getInt();
                  guardIds.push_back(guardId);
                  if (guardId == driverGuardId)
                    containsDriver = true;
                }
                if (!containsDriver)
                  continue;
                llvm::sort(guardIds);
                guardIds.erase(
                    std::unique(guardIds.begin(), guardIds.end()),
                    guardIds.end());
                SmallVector<ObservableSource, 4> pathSources;
                for (unsigned guardId : guardIds) {
                  auto srcIt = guardIdSources.find(guardId);
                  if (srcIt == guardIdSources.end())
                    continue;
                  pathSources.push_back(srcIt->second);
                }
                if (pathSources.empty())
                  continue;
                auto driverIt = llvm::find_if(
                    pathSources, [&](const ObservableSource &src) {
                      return src.takeValue == take.getResult();
                    });
                if (driverIt == pathSources.end())
                  continue;
                std::swap(pathSources.front(), *driverIt);
                if (pathSources.size() > 1)
                  llvm::sort(pathSources.begin() + 1, pathSources.end(),
                             [](const ObservableSource &a,
                                const ObservableSource &b) {
                               if (a.place != b.place)
                                 return a.place.getValue() <
                                        b.place.getValue();
                               return a.takeValue.getAsOpaquePointer() <
                                      b.takeValue.getAsOpaquePointer();
                             });
                candidateSources.push_back(std::move(pathSources));
              }
              if (!candidateSources.empty())
                usedGuardMetadata = true;
            }
          }
          if (!usedGuardMetadata) {
            if (transitionSources.empty())
              continue;
            SmallVector<ObservableSource, 4> fallbackSources(
                transitionSources.begin(), transitionSources.end());
            auto driverIt = llvm::find_if(
                fallbackSources, [&](const ObservableSource &src) {
                  return src.takeValue == take.getResult();
                });
            if (driverIt == fallbackSources.end())
              continue;
            std::swap(fallbackSources.front(), *driverIt);
            if (fallbackSources.size() > 1)
              llvm::sort(fallbackSources.begin() + 1,
                         fallbackSources.end(),
                         [](const ObservableSource &a,
                            const ObservableSource &b) {
                           if (a.place != b.place)
                             return a.place.getValue() <
                                    b.place.getValue();
                           return a.takeValue.getAsOpaquePointer() <
                                  b.takeValue.getAsOpaquePointer();
                         });
            candidateSources.push_back(std::move(fallbackSources));
          }

          for (SmallVector<ObservableSource, 4> &sources : candidateSources) {
            if (sources.empty())
              continue;
            StringAttr driver = sources.front().place;
            if (driver != driverPlace)
              continue;

            for (TargetInfo &target : targets) {
              SmallVector<ObservableSource, 4> sourcesCopy = sources;
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
      SmallVector<CursorState, 8> states;
      TokenEnv seedTokens;
      seedTokens[entry.first] = SmallVector<Value>{seed};
      for (const EdgePath &path : entry.second) {
        CursorState state;
        state.cursor = PathCursor{&path, 0, 0};
        state.token = seed;
        state.delay = Value();
        state.tokens = seedTokens;
        states.push_back(std::move(state));
      }
      if (failed(emitCursorSet(std::move(states), builder)))
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

  LogicalResult emitCursorSet(SmallVector<CursorState> states,
                              ImplicitLocOpBuilder &builder) const;
  LogicalResult emitIfGroup(ContextGroup &group,
                            ImplicitLocOpBuilder &builder) const;
  LogicalResult emitChoiceGroup(ContextGroup &group,
                                ImplicitLocOpBuilder &builder) const;
  LogicalResult emitForGroup(ContextGroup &group,
                             ImplicitLocOpBuilder &builder) const;
  LogicalResult emitLeaf(CursorState state, const EdgeTemplate *templ,
                         ImplicitLocOpBuilder &builder) const;

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

};

} // namespace



LogicalResult LPNRetainHypergraphPass::emitCursorSet(SmallVector<CursorState> states,
                            ImplicitLocOpBuilder &builder) const {
  if (states.empty())
    return success();

  SmallVector<CursorState, 4> ready;
  DenseMap<Operation *, ContextGroup> groups;
  SmallVector<Operation *, 8> order;

  for (CursorState &stateRef : states) {
    CursorState state = std::move(stateRef);
    const EdgeTemplate *templ = getTemplate(state.cursor);
    if (!templ)
      continue;
    if (state.cursor.contextIndex < templ->contexts.size()) {
      const ControlContext &ctx = templ->contexts[state.cursor.contextIndex];
      ContextGroup &group = groups[ctx.op];
      if (!group.op) {
        group.op = ctx.op;
        group.kind = ctx.kind;
        order.push_back(ctx.op);
      }
      if (ctx.kind == ContextKind::ForOp) {
        group.bodyStates.push_back(std::move(state));
      } else if (ctx.isThen) {
        group.thenStates.push_back(std::move(state));
      } else {
        group.elseStates.push_back(std::move(state));
      }
      continue;
    }
    ready.push_back(std::move(state));
  }

  for (Operation *op : order) {
    ContextGroup &group = groups[op];
    switch (group.kind) {
    case ContextKind::IfOp:
      if (failed(emitIfGroup(group, builder)))
        return failure();
      break;
    case ContextKind::ChoiceOp:
      if (failed(emitChoiceGroup(group, builder)))
        return failure();
      break;
    case ContextKind::ForOp:
      if (failed(emitForGroup(group, builder)))
        return failure();
      break;
    }
  }

  for (CursorState &state : ready) {
    const EdgeTemplate *templ = getTemplate(state.cursor);
    if (!templ)
      continue;
    if (failed(emitLeaf(std::move(state), templ, builder)))
      return failure();
  }
  return success();
}

LogicalResult LPNRetainHypergraphPass::emitIfGroup(ContextGroup &group,
                          ImplicitLocOpBuilder &builder) const {
  if (group.thenStates.empty() && group.elseStates.empty())
    return success();
  auto ifOp = dyn_cast<scf::IfOp>(group.op);
  if (!ifOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported control context");
  CursorState *rep =
      !group.thenStates.empty() ? &group.thenStates.front()
                                : &group.elseStates.front();
  const EdgeTemplate *templ = getTemplate(rep->cursor);
  if (!templ)
    return success();
  TokenEnv condTokens = rep->tokens;
  TakeEnv condTakes = rep->takes;
  if (failed(ensureTemplateSources(templ, rep->token, condTokens, condTakes,
                                   builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(rep->ssa, mapping);
  mapTemplateSources(templ, mapping, condTakes);
  Value cond = cloneValueInto(ifOp.getCondition(), mapping, builder);
  if (!cond)
    return failure();
  bool hasElse = !group.elseStates.empty();
  scf::IfOp cloned = builder.create<scf::IfOp>(TypeRange(), cond, hasElse);
  auto populate = [&](Region &region,
                      SmallVector<CursorState, 4> branchStates)
      -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                       builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (!branchStates.empty()) {
      SmallVector<CursorState, 4> advanced;
      advanced.reserve(branchStates.size());
      for (CursorState &state : branchStates) {
        CursorState next = std::move(state);
        next.cursor.contextIndex++;
        advanced.push_back(std::move(next));
      }
      if (failed(emitCursorSet(std::move(advanced), branchBuilder)))
        return failure();
    }
    branchBuilder.create<scf::YieldOp>();
    return success();
  };
  if (failed(populate(cloned.getThenRegion(), std::move(group.thenStates))))
    return failure();
  if (hasElse)
    if (failed(populate(cloned.getElseRegion(), std::move(group.elseStates))))
      return failure();
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult LPNRetainHypergraphPass::emitChoiceGroup(ContextGroup &group,
                              ImplicitLocOpBuilder &builder) const {
  if (group.thenStates.empty() && group.elseStates.empty())
    return success();
  auto choice = dyn_cast<ChoiceOp>(group.op);
  if (!choice)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported choice context");
  ChoiceOp cloned = builder.create<ChoiceOp>();
  auto populate = [&](Region &region,
                      SmallVector<CursorState, 4> branchStates)
      -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(),
                                       builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (!branchStates.empty()) {
      SmallVector<CursorState, 4> advanced;
      advanced.reserve(branchStates.size());
      for (CursorState &state : branchStates) {
        CursorState next = std::move(state);
        next.cursor.contextIndex++;
        advanced.push_back(std::move(next));
      }
      if (failed(emitCursorSet(std::move(advanced), branchBuilder)))
        return failure();
    }
    branchBuilder.create<ChoiceYieldOp>();
    return success();
  };
  if (failed(populate(cloned.getThenRegion(), std::move(group.thenStates))))
    return failure();
  if (failed(populate(cloned.getElseRegion(), std::move(group.elseStates))))
    return failure();
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult LPNRetainHypergraphPass::emitForGroup(ContextGroup &group,
                           ImplicitLocOpBuilder &builder) const {
  if (group.bodyStates.empty())
    return success();
  auto forOp = dyn_cast<scf::ForOp>(group.op);
  if (!forOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported loop context");
  CursorState &rep = group.bodyStates.front();
  const EdgeTemplate *templ = getTemplate(rep.cursor);
  if (!templ)
    return success();
  TokenEnv loopTokens = rep.tokens;
  TakeEnv loopTakes = rep.takes;
  if (failed(ensureTemplateSources(templ, rep.token, loopTokens, loopTakes,
                                   builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(rep.ssa, mapping);
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
  SmallVector<CursorState, 4> advanced;
  advanced.reserve(group.bodyStates.size());
  for (CursorState &state : group.bodyStates) {
    CursorState next = std::move(state);
    next.cursor.contextIndex++;
    next.ssa[forOp.getBody()->getArgument(0)] = cloned.getInductionVar();
    advanced.push_back(std::move(next));
  }
  if (failed(emitCursorSet(std::move(advanced), inner)))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult LPNRetainHypergraphPass::emitLeaf(CursorState state, const EdgeTemplate *templ,
                       ImplicitLocOpBuilder &builder) const {
  auto emitBody = [&](CursorState innerState,
                      ImplicitLocOpBuilder &inner) -> LogicalResult {
    if (failed(ensureTemplateSources(templ, innerState.token,
                                     innerState.tokens, innerState.takes,
                                     inner)))
      return failure();
    IRMapping mapping;
    mapContextValues(innerState.ssa, mapping);
    mapTemplateSources(templ, mapping, innerState.takes);
    Value newToken = cloneValueInto(templ->tokenValue, mapping, inner);
    if (!newToken)
      return failure();
    Value stepDelay = cloneValueInto(templ->delayValue, mapping, inner);
    if (!stepDelay)
      return failure();
    Value totalDelay = ensureDelay(innerState.delay, inner);
    totalDelay =
        inner.create<arith::AddFOp>(totalDelay, stepDelay).getResult();
    innerState.tokens[templ->target.symbol].push_back(newToken);
    bool last = innerState.cursor.path &&
                (innerState.cursor.edgeIndex + 1 ==
                 innerState.cursor.path->size());
    if (last) {
      auto placeType = PlaceType::get(inner.getContext());
      auto placeAttr = FlatSymbolRefAttr::get(templ->target.symbol);
      Value place = inner.create<PlaceRefOp>(placeType, placeAttr);
      inner.create<EmitOp>(place, newToken, totalDelay);
      return success();
    }
    CursorState next = std::move(innerState);
    next.cursor.edgeIndex++;
    next.cursor.contextIndex = 0;
    next.token = newToken;
    next.delay = totalDelay;
    SmallVector<CursorState, 1> children;
    children.push_back(std::move(next));
    return emitCursorSet(std::move(children), inner);
  };

  Value cond = buildGuardCondition(templ->target.guards, state.token, builder);
  if (!cond)
    return emitBody(std::move(state), builder);

  scf::IfOp guardIf =
      builder.create<scf::IfOp>(TypeRange(), cond, /*withElse=*/false);
  auto &guardBlock = guardIf.getThenRegion().front();
  guardBlock.clear();
  ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
  inner.setInsertionPointToStart(&guardBlock);
  if (failed(emitBody(std::move(state), inner)))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(guardIf);
  return success();
}


std::unique_ptr<Pass> createLPNRetainHypergraphPass() {
  return std::make_unique<LPNRetainHypergraphPass>();
}

} // namespace mlir::lpn
