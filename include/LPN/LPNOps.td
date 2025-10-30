#ifndef LPN_OPS
#define LPN_OPS

include "LPN/LPNDialect.td"
include "LPN/LPNTypes.td"
include "LPN/LPNAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LPN_Op<string mnemonic, list<Trait> traits = []> :
    Op<LPN_Dialect, mnemonic, traits>;

def LPN_CreateTokenOp : LPN_Op<"create_token", [Pure]> {
  // ... definition
}

// ... all other ops

#endif // LPN_OPS