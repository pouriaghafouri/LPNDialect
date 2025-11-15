set(LLVM_TARGET_DEFINITIONS LPNDialect.td)
mlir_tablegen(LPNDialect.h.inc -gen-dialect-decls -dialect=lpn)
mlir_tablegen(LPNDialect.cpp.inc -gen-dialect-defs -dialect=lpn)
add_public_tablegen_target(MLIRLPNDialectIncGen)

set(LLVM_TARGET_DEFINITIONS LPNTypes.td)
mlir_tablegen(LPNTypes.h.inc -gen-typedef-decls -typedefs-dialect=lpn)
mlir_tablegen(LPNTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=lpn)
add_public_tablegen_target(MLIRLPNTypesIncGen)

set(LLVM_TARGET_DEFINITIONS LPNAttributes.td)
mlir_tablegen(LPNAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=lpn)
mlir_tablegen(LPNAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=lpn)
add_public_tablegen_target(MLIRLPNAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS LPNOps.td)
mlir_tablegen(LPNOps.h.inc -gen-op-decls)
mlir_tablegen(LPNOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLPNOpsIncGen)
