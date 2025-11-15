#ifndef LPN_TYPES_TD
#define LPN_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "LPNDialect.td"

// A handle to a Place or Transition node in an LPN graph.
def NodeType : TypeDef<LPN_Dialect, "Node"> {
  let summary = "Handle to a place or transition node in an LPN graph";

  let description = [{
    This is a simple opaque type used as the type of SSA values produced
    by `lpn.place` and `lpn.transition` operations. Edges (`lpn.in_edge`
    and `lpn.out_edge`) take this type as operands to connect nodes in
    the Petri net graph.
  }];

  let mnemonic = "node";
}

#endif // LPN_TYPES_TD
