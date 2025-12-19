#include "LPN/Dialect/LPNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::lpn;

#include "LPN/Dialect/LPNDialect.cpp.inc"

void LPNDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "LPN/Dialect/LPNTypes.cpp.inc"
  >();
  
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LPN/Dialect/LPNAttributes.cpp.inc"
  >();
  
  addOperations<
#define GET_OP_LIST
#include "LPN/Dialect/LPNOps.cpp.inc"
  >();
}

Attribute LPNDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  parser.emitError(parser.getNameLoc())
      << "unknown attribute in LPN dialect";
  return {};
}

void LPNDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  llvm_unreachable("LPN dialect has no custom attributes");
}
