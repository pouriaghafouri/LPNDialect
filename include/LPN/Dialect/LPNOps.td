#ifndef LPN_OPS_TD
#define LPN_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "LPNDialect.td"
include "LPNTypes.td"
include "LPNAttributes.td"

class LPN_Op<string mnemonic, list<Trait> traits = []> : Op<LPN_Dialect, mnemonic, traits> {}

def LPN_Token : Type<"::mlir::lpn::TokenType">;

def CreateTokenOp : LPN_Op<"create_token", []> {
  let summary = "Creates a new LPN token";

  let description = [{
    Allocates a new token value of type !lpn.token.

    The timestamp is modeled as an i64 SSA value, not as part of the type.
    This operation accepts an optional i64 operand:

      - If no operand is provided, the runtime may assume a default
        timestamp (e.g. ts = 0).

      - If an operand is provided, that value is used as the initial
        timestamp of the token.

    Example usage:
      %t0 = lpn.create_token : !lpn.token
      %ts = arith.constant 7 : i64
      %t1 = lpn.create_token %ts : !lpn.token
  }];

  let arguments = (ins
    Optional<I64>:$timestamp
  );

  let results = (outs
    LPN_Token:$token
  );

  // Two textual forms:
  //   lpn.create_token : !lpn.token
  //   lpn.create_token %ts : !lpn.token
  let assemblyFormat =
    "($timestamp^)? attr-dict `:` type($token)";
}

def GetTimestampOp : LPN_Op<"get_timestamp", [NoSideEffect]> {
  let summary = "Reads the current timestamp associated with a token";

  let description = [{
    Extracts the timestamp associated with a token as an i64 SSA value.

    In the runtime model, this corresponds to reading tk.ts for the
    underlying Token object.

    Example usage:
    %ts = lpn.get_timestamp %t1 : !lpn.token -> i64
  }];

  let arguments = (ins
    LPN_Token:$token
  );

  let results = (outs
    I64:$timestamp
  );

  let assemblyFormat =
    "$token attr-dict `:` type($token) `->` type($timestamp)";
}

def SetTimestampOp : LPN_Op<"set_timestamp", []> {
  let summary = "Return a token with an updated timestamp";

  let description = [{
    Return a new token value that is identical to the input token,
    except that its timestamp is updated to the given i64 value.

    In SSA form, the original token value remains available; the
    updated token is produced as a new result. In the runtime model,
    this corresponds to updating tk.ts on the underlying Token object.

    example usage:

    %ts = arith.constant 42 : i64
    %t2  = lpn.set_timestamp %t1, %ts : !lpn.token -> !lpn.token
  }];

  let arguments = (ins
    LPN_Token:$token,
    I64:$new_timestamp
  );

  let results = (outs
    LPN_Token:$updated
  );

  let assemblyFormat =
    "$token `,` $new_timestamp attr-dict `:` type($token) `->` type($updated)";
}

#endif // LPN_OPS_TD
