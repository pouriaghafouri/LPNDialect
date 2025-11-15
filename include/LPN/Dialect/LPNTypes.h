#ifndef LPN_DIALECT_TYPES_H
#define LPN_DIALECT_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace lpn {
} // namespace lpn
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "LPN/Dialect/LPNTypes.h.inc"

#endif // LPN_DIALECT_TYPES_H
