#ifndef LPN_OPS_H
#define LPN_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "LPN/Dialect/LPNAttributes.h"
#include "LPN/Dialect/LPNTypes.h"

#define GET_OP_CLASSES
#include "LPN/Dialect/LPNOps.h.inc"

#endif // LPN_OPS_H