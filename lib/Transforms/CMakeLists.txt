add_mlir_library(MLIRLPNTransforms
  LPNSimulationPass.cpp
  LPNValidationPass.cpp
  LPNNormalizeDelaysPass.cpp
  LPNAbstractHiddenStatePass.cpp
  LPNSynthesizeGuardPass.cpp
  LPNStripHiddenValuesPass.cpp
  LPNLinkTokenCreatesPass.cpp
  LPNResolveChoicePass.cpp
  LPNDataflowSimplifyPass.cpp
  LPNRetainObservablesPass.cpp
  LPNRetainHypergraphPass.cpp

  PARTIAL_SOURCES_INTENDED

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
