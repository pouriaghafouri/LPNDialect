#ifndef LPN_DIALECT_H
#define LPN_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated includes
#include "LPN/LPNDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "LPN/LPNTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "LPN/LPNAttributes.h.inc"

#define GET_OP_CLASSES
#include "LPN/LPNOps.h.inc"

namespace mlir {
namespace lpn {

// Utility functions
TokenType getTokenType(MLIRContext *ctx, ArrayRef<StringRef> properties);
bool isTokenCompatibleWithPlace(TokenType tokenType, PlaceType placeType);

} // namespace lpn
} // namespace mlir

#endif // LPN_DIALECT_H