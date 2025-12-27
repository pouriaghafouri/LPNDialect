#ifndef LPN_PASSES_H
#define LPN_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace lpn {

std::unique_ptr<Pass> createLPNSimulationPass();
std::unique_ptr<Pass> createLPNValidationPass();
std::unique_ptr<Pass> createLPNNormalizeDelaysPass();
std::unique_ptr<Pass> createLPNRetainObservablesPass();
std::unique_ptr<Pass> createLPNRetainHypergraphPass();
std::unique_ptr<Pass> createLPNAbstractHiddenStatePass();
std::unique_ptr<Pass> createLPNStripHiddenValuesPass();
std::unique_ptr<Pass> createLPNSynthesizeGuardPass();
std::unique_ptr<Pass> createLPNLinkTokenCreatesPass();
std::unique_ptr<Pass> createLPNDataflowSimplifyPass();
std::unique_ptr<Pass> createLPNResolveChoicePass();
// ... other pass declarations

void registerLPNPasses();

#define GEN_PASS_REGISTRATION
#include "LPN/Conversion/LPNPasses.h.inc"

} // namespace lpn
} // namespace mlir

#endif // LPN_PASSES_H
