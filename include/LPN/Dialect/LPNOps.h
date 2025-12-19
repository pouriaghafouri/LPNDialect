#ifndef LPN_DIALECT_OPS_H
#define LPN_DIALECT_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "LPN/Dialect/LPNTypes.h"

namespace mlir {
namespace lpn {
} // namespace lpn
} // namespace mlir

#define GET_OP_CLASSES
#include "LPN/Dialect/LPNOps.h.inc"

#endif // LPN_DIALECT_OPS_H
