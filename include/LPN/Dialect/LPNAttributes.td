#ifndef LPN_ATTRIBUTES_TD
#define LPN_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "LPNDialect.td"

def TokenPropertyNamesParam : AttrParameter<"::llvm::ArrayRef<::llvm::StringRef>", "names of integer-valued token properties"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def TokenPropertySetAttr : AttrDef<LPN_Dialect, "TokenPropertySet"> {
  let summary = "schema of token properties stored in a place";

  let description = [{
    This attribute represents the logical "type" of tokens that can be
    stored in a place. Each token is a dictionary mapping these property
    names to integer values, plus an implicit integer timestamp.
  }];

  let mnemonic = "token_props";

  let parameters = (ins TokenPropertyNamesParam:$properties);
}

// Constraint so we can use it directly in ops (e.g. TokenPropertySet:$token_props).
def TokenPropertySet : Attr<"::mlir::lpn::TokenPropertySetAttr">;

#endif // LPN_ATTRIBUTES_TD
