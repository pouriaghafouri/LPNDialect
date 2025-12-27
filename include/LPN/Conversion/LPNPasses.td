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

def LPNAbstractHiddenStatePass : Pass<"lpn-abstract-hidden-state", "ModuleOp"> {
  let summary = "Annotate control flow driven by hidden places";
  let constructor = "mlir::lpn::createLPNAbstractHiddenStatePass()";
}

def LPNStripHiddenValuesPass : Pass<"lpn-strip-hidden-values", "ModuleOp"> {
  let summary = "Replace values that depend on hidden tokens with symbolic unknowns";
  let constructor = "mlir::lpn::createLPNStripHiddenValuesPass()";
}

def LPNSynthesizeGuardPass : Pass<"lpn-synthesize-guards", "ModuleOp"> {
  let summary = "Synthesize per-transition guard information.";
  let constructor = "mlir::lpn::createLPNSynthesizeGuardPass()";
}

def LPNLinkTokenCreatesPass : Pass<"lpn-link-token-creates", "ModuleOp"> {
  let summary = "Thread observable token dependencies through lpn.token.create.";
  let constructor = "mlir::lpn::createLPNLinkTokenCreatesPass()";
}

def LPNRetainObservablesPass : Pass<"lpn-retain-observables", "ModuleOp"> {
  let summary = "Collapse networks to the user-marked observable places";
  let constructor = "mlir::lpn::createLPNRetainObservablesPass()";
}

def LPNRetainHypergraphPass : Pass<"lpn-retain-hypergraph", "ModuleOp"> {
  let summary = "EXPERIMENTAL hypergraph-based observable reduction";
  let constructor = "mlir::lpn::createLPNRetainHypergraphPass()";
}

def LPNDataflowSimplifyPass : Pass<"lpn-dataflow-simplify", "ModuleOp"> {
  let summary = "Fold structurally- or effect-equivalent control flow.";
  let constructor = "mlir::lpn::createLPNDataflowSimplifyPass()";
}

def LPNResolveChoicePass : Pass<"lpn-resolve-choices", "ModuleOp"> {
  let summary = "Resolve lpn.choice nodes by assuming hidden state never blocks.";
  let constructor = "mlir::lpn::createLPNResolveChoicePass()";
}

#endif // LPN_PASSES
