#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

/// Retain Observables Pass – detailed overview
///
/// LPN authors frequently introduce many internal places (e.g., scheduler
/// queues, device FIFOs) while only caring about a handful of observable ports.
/// This pass collapses the network so that only user-marked observable places
/// remain, yet the causal relationship between observable tokens is preserved.
///
/// Steps:
///   1. **Summarize transitions.**
///      For every `lpn.transition`, walk each `lpn.take` SSA result forward to
///      find every reachable `lpn.emit`. Along the way we collect the
///      operations that mutate token logs (`TokenEdit`) and the predicates that
///      decide where the token is routed (`TokenGuard`). Each edge is stored as:
///        * `TokenEdit { field, value }` – a snapshot of `lpn.token.set`
///          instructions. For example, if a transition executes
///          `token = t.token_set(token, "hops", const 1)`, the edge records
///          `{ field = "hops", value = 1 }`.
///        * `TokenGuard { field, equalsValue }` – a minimal representation of
///          guard conditions derived from `token.get` comparisons. When we see
///          code such as `if (token.dst == 42) emit @port_42`, the guard records
///          `{ field = "dst", equalsValue = 42 }`.
///        * `ResolvedPlace { symbol, guards }` – when the emit destination is
///          constructed via `lpn.place_list.get`, we record every possible
///          symbol referenced by the list plus the guards that differentiate
///          them. For a switch with `@egress_0`, `@egress_1`, the entries look
///          like:
///          ```
///            { symbol = "tor_egress_0", guards = [{field="dst",
///                                                   equalsValue=server_start+0}] }
///            { symbol = "tor_egress_1", guards = [{field="dst",
///                                                   equalsValue=server_start+1}] }
///          ```
///      Even if transitions consume *multiple* tokens (e.g., state + request),
///      each `lpn.take` is summarized independently; this aligns with the
///      token-log semantics where every consumed token yields its own causal
///      path.
///
///   2. **Discover observable-to-observable paths.**
///      After we build the adjacency map (edges between place symbols), we run
///      a DFS starting from each observable place. Whenever the search reaches
///      another observable place we snapshot the concatenated edits + guards in
///      a `PathSummary`. This supports more involved pipelines, such as:
///        * a cache controller that consumes tokens from both `l1_req` and
///          `l2_resp`, annotates metadata, and finally emits to observable
///          `l1_resp`;
///        * queueing models where a token may pass through several scheduler
///          transitions before reaching an observable output; each intermediate
///          edge is folded into a single summary.
///
///   3. **Synthesize observable-only transitions.**
///      For every observable source place we emit a new transition that:
///        * takes one token from the source (multiple takes per transition are
///          handled by emitting multiple summaries—one per source place);
///        * applies the logged edits before each candidate emit;
///        * wraps each emit in `scf.if` blocks that replay the stored guards.
///          Because guards are re-evaluated via `token.get`, any token field
///          used in routing (e.g., `dst`, `priority`, `flow_id`) is still honored.
///
/// Limitations:
///   * Guards are currently restricted to equality against constants; more
///     complex predicates (ranges, conjunctions of multiple fields, queue-depth
///     checks via `lpn.count`, etc.) need additional pattern support before
///     they can be summarized.
///   * Token edits are captured only when their values come from
///     `arith.constant`. If a transition copies data between tokens or computes
///     edits from other SSA values, the summary will drop that path.
///   * Interactions that truly require multiple tokens simultaneously (e.g.,
///     joins that merge metadata from two inputs) are still approximated per
///     token. In practice, this matches the log semantics used in scheduler-like
///     models but may need refinement for tightly-coupled datapaths.
///
/// Despite these caveats the pass already handles transitions that consume or
/// emit multiple tokens, provided each token’s transformations are expressible
/// via SSA ops rooted at that token.

namespace mlir::lpn {
namespace {

/// Record one `lpn.token.set` effect along a path.
struct TokenEdit {
  TokenSetOp op;
};

/// Capture predicates derived from `token.get` equality checks.
struct TokenGuard {
  StringAttr field;
  int64_t equalsValue;
};

/// Edge in the place graph: source place -> target place with edits + guards.
struct EdgeSummary {
  StringAttr target;
  SmallVector<TokenEdit> edits;
  SmallVector<TokenGuard> guards;
};

/// Result of resolving a place SSA value to concrete observable symbols.
struct ResolvedPlace {
  StringAttr symbol;
  SmallVector<TokenGuard> guards;
};

/// Aggregated info for an observable-to-observable path discovered by DFS.
struct PathSummary {
  StringAttr target;
  SmallVector<TokenEdit> edits;
  SmallVector<TokenGuard> guards;
};

/// Recursively clone the SSA graph needed to materialize `value` into the
/// destination builder, using `mapping` to map previously cloned values. The
/// caller is responsible for seeding `mapping` with the current token mapping.
static Value cloneValueInto(Value value, IRMapping &mapping,
                            ImplicitLocOpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto arg = value.dyn_cast<BlockArgument>()) {
    // Block arguments are not expected in transition bodies today. Bail out.
    builder.getContext()->emitError(builder.getLoc(),
                                    "unsupported block argument in slice");
    return {};
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    builder.getContext()->emitError(builder.getLoc(),
                                    "value without defining op in slice");
    return {};
  }

  for (Value operand : def->getOperands())
    cloneValueInto(operand, mapping, builder);

  Operation *cloned = builder.clone(*def, mapping);
  return cloned->getResult(value.getResultNumber());
}

static bool editsEqual(ArrayRef<TokenEdit> lhs, ArrayRef<TokenEdit> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    const TokenEdit &a = std::get<0>(pair);
    const TokenEdit &b = std::get<1>(pair);
    if (a.op != b.op)
      return false;
  }
  return true;
}

static bool guardsEqual(ArrayRef<TokenGuard> lhs,
                        ArrayRef<TokenGuard> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    const TokenGuard &a = std::get<0>(pair);
    const TokenGuard &b = std::get<1>(pair);
    if (a.field != b.field || a.equalsValue != b.equalsValue)
      return false;
  }
  return true;
}

static void recordPath(
    StringAttr root, StringAttr target, ArrayRef<TokenEdit> edits,
    ArrayRef<TokenGuard> guards,
    DenseMap<StringAttr, SmallVector<PathSummary>> &reachable) {
  auto &entries = reachable[root];
  for (const PathSummary &existing : entries)
    if (existing.target == target && editsEqual(existing.edits, edits) &&
        guardsEqual(existing.guards, guards))
      return;
  entries.push_back(
      PathSummary{target,
                  SmallVector<TokenEdit>(edits.begin(), edits.end()),
                  SmallVector<TokenGuard>(guards.begin(), guards.end())});
}

enum class ResolveResult { Success, Unresolvable, Dynamic };

// Utility used by resolvePlaces: extract constant indices for place_list.get.
static std::optional<int64_t> tryGetConstantIndex(Value value) {
  if (auto idxConst = value.getDefiningOp<arith::ConstantIndexOp>())
    return idxConst.value();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

// Recognize index expressions based on token.get so we can rebuild guards.
static std::optional<std::pair<StringAttr, int64_t>>
matchTokenIndex(Value value) {
  Value current = value;
  if (auto castOp = current.getDefiningOp<arith::IndexCastOp>())
    current = castOp.getIn();
  if (auto subi = current.getDefiningOp<arith::SubIOp>()) {
    auto getOp = subi.getLhs().getDefiningOp<TokenGetOp>();
    auto constOp = subi.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!getOp || !constOp)
      return std::nullopt;
    if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return std::make_pair(getOp.getFieldAttr(),
                            attr.getValue().getSExtValue());
  }
  return std::nullopt;
}

// Translate a !lpn.place SSA value into one or more concrete place symbols,
// optionally annotated with token-field guards when the destination depends on
// token metadata.
static ResolveResult resolvePlaces(Value handle,
                                   SmallVectorImpl<ResolvedPlace> &placesOut) {
  if (auto ref = handle.getDefiningOp<PlaceRefOp>()) {
    ResolvedPlace entry;
    entry.symbol = ref.getPlaceAttr().getAttr();
    placesOut.push_back(entry);
    return ResolveResult::Success;
  }
  if (auto get = handle.getDefiningOp<PlaceListGetOp>()) {
    auto list = get.getList().getDefiningOp<PlaceListOp>();
    if (!list)
      return ResolveResult::Unresolvable;
    auto places = list.getPlacesAttr();
    if (auto maybeIdx = tryGetConstantIndex(get.getIndex())) {
      int64_t index = *maybeIdx;
      if (index < 0 || index >= static_cast<int64_t>(places.size()))
        return ResolveResult::Unresolvable;
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(places[index])) {
        ResolvedPlace entry;
        entry.symbol = sym.getAttr();
        placesOut.push_back(entry);
        return ResolveResult::Success;
      }
      return ResolveResult::Unresolvable;
    }
    auto guardInfo = matchTokenIndex(get.getIndex());
    if (!guardInfo)
      return ResolveResult::Dynamic;
    auto [field, offset] = *guardInfo;
    for (auto [idx, attr] : llvm::enumerate(places)) {
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(attr)) {
        ResolvedPlace entry;
        entry.symbol = sym.getAttr();
        entry.guards.push_back(TokenGuard{field, offset + static_cast<int64_t>(idx)});
        placesOut.push_back(entry);
      }
    }
    return placesOut.empty() ? ResolveResult::Unresolvable
                             : ResolveResult::Success;
  }
  return ResolveResult::Unresolvable;
}

// Track how a taken token flows through token.set/clone operations and collect
// every reachable emit operation along with the edits that occurred beforehand.
static bool collectTokenEdits(
    Value value, SmallVectorImpl<TokenEdit> &current,
    SmallVectorImpl<std::pair<EmitOp, SmallVector<TokenEdit>>> &paths) {
  for (Operation *user : value.getUsers()) {
    if (auto emit = dyn_cast<EmitOp>(user)) {
      SmallVector<TokenEdit> snapshot(current.begin(), current.end());
      paths.emplace_back(emit, snapshot);
      continue;
    }
    if (auto set = dyn_cast<TokenSetOp>(user)) {
      current.emplace_back(TokenEdit{set});
      if (!collectTokenEdits(set.getResult(), current, paths))
        return false;
      current.pop_back();
      continue;
    }
    if (auto clone = dyn_cast<TokenCloneOp>(user)) {
      if (!collectTokenEdits(clone.getResult(), current, paths))
        return false;
      continue;
    }
    if (isa<TokenGetOp>(user))
      continue;
    return false;
  }
  return true;
}

// DFS over the summarized graph to build observable-to-observable path
// summaries (token edits + guard predicates).
static void accumulatePaths(
    StringAttr root, StringAttr current, DenseSet<StringAttr> &visited,
    SmallVector<TokenEdit> &pendingEdits,
    SmallVector<TokenGuard> &pendingGuards,
    const DenseSet<StringAttr> &observableNames,
    DenseMap<StringAttr, SmallVector<PathSummary>> &reachable,
    const DenseMap<StringAttr, SmallVector<EdgeSummary>> &adjacency) {
  if (observableNames.contains(current) && current != root) {
    recordPath(root, current, pendingEdits, pendingGuards, reachable);
    return;
  }

  auto it = adjacency.find(current);
  if (it == adjacency.end())
    return;

  for (const EdgeSummary &edge : it->second) {
    size_t priorEdits = pendingEdits.size();
    size_t priorGuards = pendingGuards.size();
    pendingEdits.append(edge.edits.begin(), edge.edits.end());
    pendingGuards.append(edge.guards.begin(), edge.guards.end());
    if (observableNames.contains(edge.target)) {
      recordPath(root, edge.target, pendingEdits, pendingGuards, reachable);
      pendingEdits.resize(priorEdits);
      pendingGuards.resize(priorGuards);
      continue;
    }
    if (visited.insert(edge.target).second) {
      accumulatePaths(root, edge.target, visited, pendingEdits, pendingGuards,
                      observableNames, reachable, adjacency);
      visited.erase(edge.target);
    }
    pendingEdits.resize(priorEdits);
    pendingGuards.resize(priorGuards);
  }
}

struct LPNRetainObservablesPass
    : PassWrapper<LPNRetainObservablesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNRetainObservablesPass)

  StringRef getArgument() const final { return "lpn-retain-observables"; }
  StringRef getDescription() const final {
    return "Collapse the network down to user-marked observable places.";
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
    MLIRContext *ctx = net.getContext();
    ctx->loadDialect<scf::SCFDialect>();
    SmallVector<PlaceOp> observablePlaces;
    for (Operation &op : net.getBody().getOps()) {
      if (auto place = dyn_cast<PlaceOp>(op)) {
        if (place.getObservableAttr())
          observablePlaces.push_back(place);
      }
    }

    if (observablePlaces.size() < 2)
      return success();

    DenseMap<StringAttr, SmallVector<EdgeSummary>> adjacency;
    SmallVector<TransitionOp> originalTransitions;
    bool sawDynamicRouting = false;
    for (TransitionOp trans : net.getOps<TransitionOp>()) {
      originalTransitions.push_back(trans);
      for (TakeOp take :
           llvm::make_early_inc_range(trans.getBody().getOps<TakeOp>())) {
        SmallVector<ResolvedPlace> sources;
        switch (resolvePlaces(take.getPlace(), sources)) {
        case ResolveResult::Success:
          break;
        case ResolveResult::Dynamic:
          sawDynamicRouting = true;
          [[fallthrough]];
        case ResolveResult::Unresolvable:
          continue;
        }
        SmallVector<std::pair<EmitOp, SmallVector<TokenEdit>>> paths;
        SmallVector<TokenEdit> current;
        if (!collectTokenEdits(take.getResult(), current, paths))
          continue;
        for (auto &[emit, edits] : paths) {
          SmallVector<ResolvedPlace> targets;
          switch (resolvePlaces(emit.getPlace(), targets)) {
          case ResolveResult::Success:
            break;
          case ResolveResult::Dynamic:
            sawDynamicRouting = true;
            [[fallthrough]];
          case ResolveResult::Unresolvable:
            continue;
          }
          for (const ResolvedPlace &src : sources) {
            for (const ResolvedPlace &dst : targets) {
              EdgeSummary edge;
              edge.target = dst.symbol;
              edge.edits = SmallVector<TokenEdit>(edits.begin(), edits.end());
              edge.guards = dst.guards;
              adjacency[src.symbol].push_back(edge);
            }
          }
        }
      }
    }

    if (sawDynamicRouting) {
      net.emitRemark(
          "retain-observables skipped: encountered dynamic place selection");
      return success();
    }

    DenseSet<StringAttr> observableNames;
    for (PlaceOp place : observablePlaces)
      observableNames.insert(place.getSymNameAttr());

    DenseMap<StringAttr, SmallVector<PathSummary>> reachable;
    for (PlaceOp place : observablePlaces) {
      StringAttr src = place.getSymNameAttr();
      DenseSet<StringAttr> visited;
      visited.insert(src);
      SmallVector<TokenEdit> pending;
      SmallVector<TokenGuard> guardStack;
      accumulatePaths(src, src, visited, pending, guardStack, observableNames,
                      reachable, adjacency);
    }

    if (reachable.empty())
      return success();

    HaltOp halt = nullptr;
    for (Operation &op : net.getBody().getOps())
      if ((halt = dyn_cast<HaltOp>(op)))
        break;

    if (!halt)
      return net.emitError("missing lpn.halt");

    Block &block = net.getBody().front();
    OpBuilder builder(&block, Block::iterator(halt));
    int summaryIndex = 0;
    for (auto &entry : reachable) {
      std::string name = (entry.first.getValue() + "_summary").str();
      if (name.empty())
        name = "summary";
      name.append("_");
      name.append(std::to_string(summaryIndex++));

      auto trans =
          builder.create<TransitionOp>(net.getLoc(), builder.getStringAttr(name));
      Region &region = trans.getBody();
      Block *body = new Block();
      region.push_back(body);
      ImplicitLocOpBuilder bodyBuilder(net.getLoc(), ctx);
      bodyBuilder.setInsertionPointToStart(body);
      auto srcRef = bodyBuilder.create<PlaceRefOp>(
          PlaceType::get(ctx), FlatSymbolRefAttr::get(entry.first));
      Value baseToken =
          bodyBuilder.create<TakeOp>(TokenType::get(ctx), srcRef.getResult())
              .getResult();
      auto zero = bodyBuilder.create<arith::ConstantOp>(
          bodyBuilder.getF64FloatAttr(0.0));
      auto i64Ty = IntegerType::get(ctx, 64);

      for (PathSummary &path : entry.second) {
        auto dstRef = bodyBuilder.create<PlaceRefOp>(
            PlaceType::get(ctx), FlatSymbolRefAttr::get(path.target));
        IRMapping mapping;
        mapping.map(takeValueMap[entry.first], baseToken); // Wait: need mapping? we don't have take map.
        for (const TokenEdit &edit : path.edits) {
          auto typed = cast<TypedAttr>(edit.value);
          auto value = bodyBuilder.create<arith::ConstantOp>(typed);
          pathToken = bodyBuilder
                          .create<TokenSetOp>(TokenType::get(ctx), pathToken,
                                              edit.field, value)
                          .getResult();
        }

        if (path.guards.empty()) {
          bodyBuilder.create<EmitOp>(dstRef.getResult(), pathToken, zero);
          continue;
        }

        Value cond;
        for (const TokenGuard &guard : path.guards) {
          Value observed =
              bodyBuilder.create<TokenGetOp>(i64Ty, baseToken, guard.field);
          auto expected = bodyBuilder.create<arith::ConstantOp>(
              bodyBuilder.getI64IntegerAttr(guard.equalsValue));
          Value eq = bodyBuilder.create<arith::CmpIOp>(
              arith::CmpIPredicate::eq, observed, expected);
          cond = cond ? bodyBuilder.create<arith::AndIOp>(cond, eq) : eq;
        }
        auto ifOp = bodyBuilder.create<scf::IfOp>(TypeRange(), cond, false);
        Block &thenBlock = ifOp.getThenRegion().front();
        OpBuilder thenBuilder(ctx);
        thenBuilder.setInsertionPoint(thenBlock.getTerminator());
        thenBuilder.create<EmitOp>(net.getLoc(), dstRef.getResult(), pathToken,
                                   zero);
      }
      bodyBuilder.create<ScheduleReturnOp>();
    }

    SmallVector<Operation *> toErase;
    for (PlaceOp place : net.getOps<PlaceOp>()) {
      if (!place.getObservableAttr())
        toErase.push_back(place);
    }
    for (TransitionOp trans : originalTransitions)
      toErase.push_back(trans);
    for (Operation *op : toErase)
      op->erase();

    return success();
  }
};

}  // namespace

std::unique_ptr<Pass> createLPNRetainObservablesPass() {
  return std::make_unique<LPNRetainObservablesPass>();
}

}  // namespace mlir::lpn
