#include "LPN/LPNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::lpn;

#include "LPN/LPNDialect.cpp.inc"

void LPNDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "LPN/LPNTypes.cpp.inc"
  >();
  
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LPN/LPNAttributes.cpp.inc"
  >();
  
  addOperations<
#define GET_OP_LIST
#include "LPN/LPNOps.cpp.inc"
  >();
}