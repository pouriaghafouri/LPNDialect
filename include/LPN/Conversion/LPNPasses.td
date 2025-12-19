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

def LPNValidationPass : Pass<"lpn-validate", "ModuleOp"> {
  let summary = "Validate structural invariants of an LPN module";
  let constructor = "mlir::lpn::createLPNValidationPass()";
}

def LPNNormalizeDelaysPass : Pass<"lpn-normalize-delays", "ModuleOp"> {
  let summary = "Canonicalize emit delays to enable downstream analyses";
  let constructor = "mlir::lpn::createLPNNormalizeDelaysPass()";
}

def LPNFusePrivatePlacesPass : Pass<"lpn-fuse-private-places", "ModuleOp"> {
  let summary = "Stub pass for collapsing transitions connected via private places";
  let constructor = "mlir::lpn::createLPNFusePrivatePlacesPass()";
}

def LPNAbstractHiddenStatePass : Pass<"lpn-abstract-hidden-state", "ModuleOp"> {
  let summary = "Annotate control flow driven by hidden places";
  let constructor = "mlir::lpn::createLPNAbstractHiddenStatePass()";
}

def LPNRetainObservablesPass : Pass<"lpn-retain-observables", "ModuleOp"> {
  let summary = "Collapse networks to the user-marked observable places";
  let constructor = "mlir::lpn::createLPNRetainObservablesPass()";
}

#endif // LPN_PASSES
