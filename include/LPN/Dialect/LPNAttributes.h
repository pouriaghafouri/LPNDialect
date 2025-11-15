#ifndef LPN_DIALECT_ATTRIBUTES_H
#define LPN_DIALECT_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace lpn {
} // namespace lpn
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "LPN/Dialect/LPNAttributes.h.inc"

#endif // LPN_DIALECT_ATTRIBUTES_H
