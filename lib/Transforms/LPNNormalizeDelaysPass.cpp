#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

namespace mlir::lpn {
namespace {

struct LPNNormalizeDelaysPass
    : PassWrapper<LPNNormalizeDelaysPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNNormalizeDelaysPass)

  StringRef getArgument() const final { return "lpn-normalize-delays"; }
  StringRef getDescription() const final {
    return "Rewrite lpn.emit ops to use a canonical zero-delay operand.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    module.walk([&](lpn::EmitOp emit) {
      Value delay = emit.getDelay();
      if (auto cst = delay.getDefiningOp<arith::ConstantOp>()) {
        if (auto floatAttr = dyn_cast<FloatAttr>(cst.getValue()))
          if (floatAttr.getValue().isZero())
            return;
      }
      builder.setInsertionPoint(emit);
      auto zeroAttr = builder.getF64FloatAttr(0.0);
      ImplicitLocOpBuilder locBuilder(emit.getLoc(), builder);
      auto zero = locBuilder.create<arith::ConstantOp>(zeroAttr);
      emit.getDelayMutable().assign(zero);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createLPNNormalizeDelaysPass() {
  return std::make_unique<LPNNormalizeDelaysPass>();
}

}  // namespace mlir::lpn
