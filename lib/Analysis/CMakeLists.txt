add_mlir_library(MLIRLPNAnalysis
  DeadlockAnalysis.cpp
  ReachabilityAnalysis.cpp
  PerformanceAnalysis.cpp
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLPN
)