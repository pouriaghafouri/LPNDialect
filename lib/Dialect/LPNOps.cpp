#include "LPN/Dialect/LPNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::lpn;

#define GET_OP_CLASSES
#include "LPN/Dialect/LPNOps.cpp.inc"
