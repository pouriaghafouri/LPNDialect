set(LLVM_TARGET_DEFINITIONS LPNPasses.td)
mlir_tablegen(LPNPasses.h.inc -gen-pass-decls -name LPN)
add_public_tablegen_target(MLIRLPNPassIncGen)
