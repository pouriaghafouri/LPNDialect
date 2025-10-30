#ifndef LPN_PASSES_H
#define LPN_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace lpn {

std::unique_ptr<Pass> createLPNSimulationPass();
std::unique_ptr<Pass> createLPNValidationPass();
// ... other pass declarations

void registerLPNPasses();

#define GEN_PASS_REGISTRATION
#include "LPN/LPNPasses.h.inc"

} // namespace lpn
} // namespace mlir

#endif // LPN_PASSES_H