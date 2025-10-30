#ifndef LPN_DIALECT
#define LPN_DIALECT

include "mlir/IR/OpBase.td"

def LPN_Dialect : Dialect {
  let name = "lpn";
  let summary = "Latency Petri Net dialect";
  let description = [{...}];
  let cppNamespace = "::mlir::lpn";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // LPN_DIALECT