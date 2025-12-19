#include "LPN/Dialect/LPNTypes.h"
#include "LPN/Dialect/LPNDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::lpn;

#define GET_TYPEDEF_CLASSES
#include "LPN/Dialect/LPNTypes.cpp.inc"
