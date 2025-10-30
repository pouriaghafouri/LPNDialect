#include "LPN/LPNDialect.h"
#include "LPN/LPNPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::lpn::registerLPNPasses();
  
  DialectRegistry registry;
  registry.insert<mlir::lpn::LPNDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LPN optimizer\n", registry));
}