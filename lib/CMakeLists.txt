add_subdirectory(Transforms)
add_subdirectory(Analysis)

add_mlir_dialect_library(MLIRLPN
  LPNDialect.cpp
  LPNTypes.cpp
  LPNAttributes.cpp
  LPNOps.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/LPN
  
  DEPENDS
  MLIRLPNIncGen
  MLIRLPNPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLPNTransforms
  MLIRLPNAnalysis
)