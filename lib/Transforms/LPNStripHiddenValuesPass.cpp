#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Pass/Pass.h"

namespace mlir::lpn {
namespace {

static LogicalResult resolvePlaceSymbol(Value handle, StringAttr &symbol) {
  auto ref = handle.getDefiningOp<PlaceRefOp>();
  if (!ref)
    return failure();
  symbol = ref.getPlaceAttr().getAttr();
  return success();
}

static Value buildUnknownValue(Type type, Location loc,
                               OpBuilder &builder) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    auto attr = builder.getIntegerAttr(intTy, 0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    auto attr = builder.getFloatAttr(floatTy, 0.0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  if (isa<IndexType>(type)) {
    auto attr = builder.getIndexAttr(0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  return Value();
}

/// Note: this pass is conservative.  Once a token originates from a non
/// observable place we treat all of its fields as unknown, even if earlier
/// operations copied observable data into the token.  This allows later passes
/// to reason about hidden-state independence, but it also means we currently
/// lose precision for hidden tokens that merely relay observable metadata.
struct LPNStripHiddenValuesPass
    : PassWrapper<LPNStripHiddenValuesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNStripHiddenValuesPass)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final {
    return "lpn-strip-hidden-values";
  }

  StringRef getDescription() const final {
    return "Replace values depending on hidden tokens with symbolic unknowns.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::DenseSet<Value> hiddenTokens;

    module.walk([&](NetOp net) {
      llvm::DenseSet<StringAttr> observable;
      for (PlaceOp place : net.getOps<PlaceOp>())
        if (place.getObservableAttr())
          observable.insert(place.getSymNameAttr());
      net.walk([&](TransitionOp trans) {
        SmallVector<TakeOp, 8> takes;
        trans.walk([&](TakeOp take) { takes.push_back(take); });
        for (TakeOp take : llvm::make_early_inc_range(takes)) {
          StringAttr symbol;
          if (failed(resolvePlaceSymbol(take.getPlace(), symbol)))
            continue;
          if (observable.contains(symbol))
            continue;
          OpBuilder builder(take);
          auto token =
              builder.create<TokenCreateOp>(take.getLoc(),
                                            take.getResult().getType(),
                                            ValueRange{}, nullptr);
          hiddenTokens.insert(token.getResult());
          take.getResult().replaceAllUsesWith(token.getResult());
          take.erase();
        }
      });
    });

    SmallVector<Value, 8> worklist(hiddenTokens.begin(), hiddenTokens.end());
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      for (Operation *user : current.getUsers()) {
        if (auto clone = dyn_cast<TokenCloneOp>(user)) {
          if (hiddenTokens.insert(clone.getResult()).second)
            worklist.push_back(clone.getResult());
          continue;
        }
        if (auto set = dyn_cast<TokenSetOp>(user)) {
          if (hiddenTokens.insert(set.getResult()).second)
            worklist.push_back(set.getResult());
        }
      }
    }

    WalkResult walkResult = module.walk([&](TokenGetOp get) -> WalkResult {
      if (!hiddenTokens.contains(get.getToken()))
        return WalkResult::advance();
      OpBuilder builder(get);
      Value replacement =
          buildUnknownValue(get.getResult().getType(), get.getLoc(), builder);
      if (!replacement) {
        get.emitError("unsupported hidden value type for stripping");
        return WalkResult::interrupt();
      }
      get.getResult().replaceAllUsesWith(replacement);
      get.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNStripHiddenValuesPass() {
  return std::make_unique<LPNStripHiddenValuesPass>();
}

} // namespace mlir::lpn
