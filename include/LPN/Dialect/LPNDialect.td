#ifndef LPN_DIALECT_TD
#define LPN_DIALECT_TD

include "mlir/IR/DialectBase.td"

def LPN_Dialect : Dialect {
  let name = "lpn";

  let cppNamespace = "::mlir::lpn";

  let summary = "Latency Petri Net Dialect";

  let description = [{
    The LPN dialect models Latency Petri Nets. A Petri net consists of
    places (which hold tokens) and transitions (which move tokens with
    delay, weight, and guard semantics).
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let usePropertiesForAttributes = 1;
}

#endif // LPN_DIALECT_TD
