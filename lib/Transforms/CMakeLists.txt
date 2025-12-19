add_mlir_library(MLIRLPNTransforms
  LPNSimulationPass.cpp
  LPNValidationPass.cpp
  LPNNormalizeDelaysPass.cpp
  LPNFusePrivatePlacesPass.cpp
  LPNAbstractHiddenStatePass.cpp
  LPNRetainObservablesPass.cpp

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
