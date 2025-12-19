add_llvm_executable(lpn-opt
  lpn-opt.cpp
)

target_link_libraries(lpn-opt PRIVATE
  MLIRLPN
  MLIRLPNTransforms
  MLIROptLib
  MLIRTransforms
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)
