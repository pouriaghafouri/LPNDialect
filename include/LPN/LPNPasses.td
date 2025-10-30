#ifndef LPN_PASSES
#define LPN_PASSES

include "mlir/Pass/PassBase.td"

def LPNSimulationPass : Pass<"lpn-simulate", "ModuleOp"> {
  let summary = "Simulate LPN networks";
  let constructor = "mlir::lpn::createLPNSimulationPass()";
  let options = [
    Option<"maxTime", "max-time", "int64_t", "1000", "Max simulation time">
  ];
}

// ... other passes

#endif // LPN_PASSES