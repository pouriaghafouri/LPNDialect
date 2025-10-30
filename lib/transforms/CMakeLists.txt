add_mlir_library(MLIRLPNTransforms
  LPNSimulationPass.cpp
  LPNValidationPass.cpp
  LPNOptimizationPass.cpp
  LPNToPI.cpp
  LPNToSMT.cpp
  LPNToLLVM.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  
  DEPENDS
  MLIRLPNPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRLPN
)