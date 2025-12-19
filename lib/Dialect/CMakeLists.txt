add_mlir_dialect_library(MLIRLPN
  LPNDialect.cpp
  LPNOps.cpp
  LPNTypes.cpp
  LPNAttributes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/LPN

  DEPENDS
  MLIRLPNDialectIncGen
  MLIRLPNTypesIncGen
  MLIRLPNAttributesIncGen
  MLIRLPNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
