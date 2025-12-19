#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::lpn {
namespace {

/// Determine if a value depends on any tokens originating from hidden places.
static bool dependsOnHiddenTokens(Value value,
                                  const DenseSet<Value> &hiddenTokens,
                                  SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (hiddenTokens.contains(value))
    return true;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  for (Value operand : def->getOperands())
    if (dependsOnHiddenTokens(operand, hiddenTokens, visited))
      return true;
  return false;
}

static std::optional<StringAttr> resolvePlaceSymbol(Value handle) {
  if (auto ref = handle.getDefiningOp<PlaceRefOp>())
    return ref.getPlaceAttr().getAttr();
  return std::nullopt;
}

static void propagateHiddenTokens(Block &block,
                                  const DenseSet<StringAttr> &observablePlaces,
                                  DenseSet<Value> &hiddenTokens) {
  for (Operation &op : block) {
    if (auto take = dyn_cast<TakeOp>(op)) {
      if (auto sym = resolvePlaceSymbol(take.getPlace()))
        if (!observablePlaces.contains(*sym))
          hiddenTokens.insert(take.getResult());
      continue;
    }
    if (auto set = dyn_cast<TokenSetOp>(op)) {
      if (hiddenTokens.contains(set.getToken()))
        hiddenTokens.insert(set.getResult());
      continue;
    }
    if (auto clone = dyn_cast<TokenCloneOp>(op)) {
      if (hiddenTokens.contains(clone.getToken()))
        hiddenTokens.insert(clone.getResult());
      continue;
    }
    if (auto get = dyn_cast<TokenGetOp>(op)) {
      if (hiddenTokens.contains(get.getToken()))
        hiddenTokens.insert(get.getResult());
      continue;
    }
  }
}

struct LPNAbstractHiddenStatePass
    : PassWrapper<LPNAbstractHiddenStatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNAbstractHiddenStatePass)

  StringRef getArgument() const final {
    return "lpn-abstract-hidden-state";
  }
  StringRef getDescription() const final {
    return "Annotate control flow driven by hidden places so downstream passes "
           "can treat it as symbolic choice.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](NetOp net) {
      DenseSet<StringAttr> observablePlaces;
      for (PlaceOp place : net.getOps<PlaceOp>())
        if (place.getObservableAttr())
          observablePlaces.insert(place.getSymNameAttr());
      Builder builder(net.getContext());
      net.walk([&](TransitionOp trans) {
        DenseSet<Value> hiddenTokens;
        propagateHiddenTokens(trans.getBody().front(), observablePlaces,
                              hiddenTokens);
        for (Operation &op : trans.getBody().front()) {
          if (auto get = dyn_cast<TokenGetOp>(op)) {
            if (!hiddenTokens.contains(get.getResult()))
              continue;
            get->setAttr("lpn.hidden_value", builder.getUnitAttr());
          }
        }
        trans.walk([&](scf::IfOp ifOp) {
          SmallPtrSet<Value, 8> visited;
          if (!dependsOnHiddenTokens(ifOp.getCondition(), hiddenTokens,
                                     visited))
            return;
          if (ifOp->hasAttr("lpn.hidden_choice"))
            return;
          ifOp->setAttr("lpn.hidden_choice", builder.getUnitAttr());
        });
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> createLPNAbstractHiddenStatePass() {
  return std::make_unique<LPNAbstractHiddenStatePass>();
}

} // namespace mlir::lpn
