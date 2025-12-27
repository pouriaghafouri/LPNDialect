#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::lpn {
namespace {

static LogicalResult resolvePlace(Value handle, StringAttr &symbol) {
  if (auto ref = handle.getDefiningOp<PlaceRefOp>()) {
    symbol = ref.getPlaceAttr().getAttr();
    return success();
  }
  return failure();
}

static void rewriteCreate(TokenCreateOp create,
                          ArrayRef<Value> dependencyTokens) {
  if (dependencyTokens.empty())
    return;
  OpBuilder builder(create);
  auto newOp =
      builder.create<TokenCreateOp>(create.getLoc(), create.getResult().getType(),
                                    dependencyTokens, create.getLogPrefixAttr());
  newOp->setAttrs(create->getAttrs());
  create.replaceAllUsesWith(newOp.getResult());
  create.erase();
}

static void visitRegion(
    Region &region, const DenseSet<StringAttr> &observablePlaces,
    SmallVector<Value, 8> activeTokens,
    llvm::SmallDenseSet<Value, 8> activeSet);

static void visitBlock(Block &block,
                       const DenseSet<StringAttr> &observablePlaces,
                       SmallVector<Value, 8> &activeTokens,
                       llvm::SmallDenseSet<Value, 8> &activeSet) {
  for (auto it = block.begin(), e = block.end(); it != e;) {
    Operation &op = *it++;

    if (auto take = dyn_cast<TakeOp>(op)) {
      StringAttr symbol;
      if (succeeded(resolvePlace(take.getPlace(), symbol)) &&
          observablePlaces.contains(symbol)) {
        if (activeSet.insert(take.getResult()).second)
          activeTokens.push_back(take.getResult());
      }
      continue;
    }

    if (auto create = dyn_cast<TokenCreateOp>(op)) {
      rewriteCreate(create, activeTokens);
      continue;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Value, 8> thenTokens = activeTokens;
      llvm::SmallDenseSet<Value, 8> thenSet = activeSet;
      visitRegion(ifOp.getThenRegion(), observablePlaces, thenTokens, thenSet);

      if (!ifOp.getElseRegion().empty()) {
        SmallVector<Value, 8> elseTokens = activeTokens;
        llvm::SmallDenseSet<Value, 8> elseSet = activeSet;
        visitRegion(ifOp.getElseRegion(), observablePlaces, elseTokens, elseSet);
      }
      continue;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      SmallVector<Value, 8> bodyTokens = activeTokens;
      llvm::SmallDenseSet<Value, 8> bodySet = activeSet;
      visitRegion(forOp.getRegion(), observablePlaces, bodyTokens, bodySet);
      continue;
    }
  }
}

static void visitRegion(
    Region &region, const DenseSet<StringAttr> &observablePlaces,
    SmallVector<Value, 8> activeTokens,
    llvm::SmallDenseSet<Value, 8> activeSet) {
  if (region.empty())
    return;
  visitBlock(region.front(), observablePlaces, activeTokens, activeSet);
}

struct LPNLinkTokenCreatesPass
    : PassWrapper<LPNLinkTokenCreatesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNLinkTokenCreatesPass)

  StringRef getArgument() const final { return "lpn-link-token-creates"; }
  StringRef getDescription() const final {
    return "Thread observable take dependencies through lpn.token.create ops.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](NetOp net) {
      DenseSet<StringAttr> observablePlaces;
      for (PlaceOp place : net.getOps<PlaceOp>())
        if (place.getObservableAttr())
          observablePlaces.insert(place.getSymNameAttr());
      net.walk([&](TransitionOp trans) {
        SmallVector<Value, 8> activeTokens;
        llvm::SmallDenseSet<Value, 8> activeSet;
        visitBlock(trans.getBody().front(), observablePlaces, activeTokens,
                   activeSet);
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> createLPNLinkTokenCreatesPass() {
  return std::make_unique<LPNLinkTokenCreatesPass>();
}

} // namespace mlir::lpn
