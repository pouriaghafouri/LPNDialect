#ifndef LPN_TYPES_TD
#define LPN_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "LPNDialect.td"

def TokenType : TypeDef<LPN_Dialect, "Token"> {
  let summary = "Opaque handle representing a token consumed/produced by schedules";

  let description = [{
    Tokens are abstract values that flow through schedule functions. They can
    only be materialized by `lpn.take` and consumed by `lpn.emit` or the other
    token-manipulation operations provided by this dialect.
  }];

  let mnemonic = "token";
}

def PlaceHandleType : TypeDef<LPN_Dialect, "Place"> {
  let summary = "Handle to a place";

  let description = [{
    Represents a runtime reference to a place. Obtain handles via
    `lpn.place_ref` or by indexing into place lists.
  }];

  let mnemonic = "place";
}

#endif // LPN_TYPES_TD
