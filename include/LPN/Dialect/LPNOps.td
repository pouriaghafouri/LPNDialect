#ifndef LPN_OPS_TD
#define LPN_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "LPNDialect.td"
include "LPNTypes.td"
include "LPNAttributes.td"

class LPN_Op<string mnemonic, list<Trait> traits = []> : Op<LPN_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Structural operations
//===----------------------------------------------------------------------===//

def NetOp : LPN_Op<"net", [
    SymbolTable,
    SingleBlockImplicitTerminator<"mlir::lpn::HaltOp">,
    IsolatedFromAbove
  ]> {
  let summary = "Root container for an LPN network";

  let description = [{
    Encapsulates a scheduler-driven Petri net. The region holds `lpn.place`
    declarations followed by `lpn.transition` definitions and a final
    `lpn.halt`. The op establishes a local symbol table so that nested
    operations may refer to places and transitions by name.
  }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "attr-dict-with-keyword $body";
}

def PlaceOp : LPN_Op<"place", [Symbol]> {
  let summary = "Declare a place";

  let description = [{
    Declares a place within an `lpn.net`. Places may optionally specify a
    capacity and/or initial token count.  When the optional `observable`
    attribute is present the place marks an external interface point that must
    be preserved by downstream reduction passes.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<I64Attr>:$initial_tokens,
    OptionalAttr<UnitAttr>:$observable
  );

  let assemblyFormat = "$sym_name attr-dict";
}

def HaltOp : LPN_Op<"halt", [Terminator]> {
  let summary = "Terminate an `lpn.net` body";
  let assemblyFormat = "attr-dict";
}

def TransitionOp : LPN_Op<"transition", [
    Symbol,
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"mlir::lpn::ScheduleReturnOp">
  ]> {
  let summary = "Define a scheduler body";

  let description = [{
    Models a scheduler (transition) responsible for manipulating tokens across
    places. The transition body is a single-block region terminated with
    `lpn.schedule.return`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def ScheduleReturnOp : LPN_Op<"schedule.return", [Terminator]> {
  let summary = "Return from a transition region";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Place handle utilities
//===----------------------------------------------------------------------===//

def PlaceRefOp : LPN_Op<"place_ref", [Pure]> {
  let summary = "Materialize a place handle";

  let arguments = (ins
    FlatSymbolRefAttr:$place
  );

  let results = (outs
    PlaceHandleType:$handle
  );

  let assemblyFormat =
    "$place attr-dict `:` type($handle)";
}

def PlaceListOp : LPN_Op<"place_list", [Pure]> {
  let summary = "Create a list of place handles";

  let arguments = (ins
    FlatSymbolRefArrayAttr:$places
  );

  let results = (outs
    PlaceListType:$list
  );

  let assemblyFormat =
    "attr-dict `:` type($list)";
}

def PlaceListGetOp : LPN_Op<"place_list.get", [Pure]> {
  let summary = "Read a handle from a place list";

  let arguments = (ins
    PlaceListType:$list,
    Index:$index
  );

  let results = (outs
    PlaceHandleType:$handle
  );

  let assemblyFormat =
    "$list `,` $index attr-dict `:` `(` type($list) `,` type($index) `)` `->` type($handle)";
}

//===----------------------------------------------------------------------===//
// Token flow primitives
//===----------------------------------------------------------------------===//

def TakeOp : LPN_Op<"take"> {
  let summary = "Remove a token from a place";

  let arguments = (ins
    PlaceHandleType:$place
  );

  let results = (outs
    TokenType:$token
  );

  let assemblyFormat =
    "$place attr-dict `:` type($place) `->` type($token)";
}

def EmitOp : LPN_Op<"emit"> {
  let summary = "Schedule a token into a place after a delay";

  let arguments = (ins
    PlaceHandleType:$place,
    TokenType:$token,
    F64:$delay
  );

  let assemblyFormat =
    "$place `,` $token `,` $delay attr-dict `:` type($place) `,` type($token) `,` type($delay)";
}

def CountOp : LPN_Op<"count", [Pure]> {
  let summary = "Read the current occupancy of a place";

  let arguments = (ins
    PlaceHandleType:$place
  );

  let results = (outs
    I64:$count
  );

  let assemblyFormat =
    "$place attr-dict `:` type($place) `->` type($count)";
}

//===----------------------------------------------------------------------===//
// Token utilities
//===----------------------------------------------------------------------===//

def TokenCreateOp : LPN_Op<"token.create", [Pure]> {
  let summary = "Create a token with an optional literal property log";

  let description = [{
    Allocate a fresh token handle of type `!lpn.token`.  Tokens are modeled as
    append-only logs of `(key, value)` pairs.  The optional `log_prefix`
    attribute allows the IR to materialize a compile-time log prefix (each
    dictionary entry is appended in attribute iteration order).  If omitted, the
    token starts with an empty log.
  }];

  let arguments = (ins
    OptionalAttr<DictionaryAttr>:$log_prefix
  );

  let results = (outs
    TokenType:$token
  );

  let assemblyFormat =
    "($log_prefix^)? attr-dict `:` type($token)";
}

def TokenCloneOp : LPN_Op<"token.clone", [Pure]> {
  let summary = "Duplicate a token handle";

  let description = [{
    Produce a second SSA value that references the same runtime token.  Since
    updates are modeled as new log entries (not destructive mutation), cloning
    is a trivial, side-effect-free operation.
  }];

  let arguments = (ins
    TokenType:$token
  );

  let results = (outs
    TokenType:$copy
  );

  let assemblyFormat =
    "$token attr-dict `:` type($token) `->` type($copy)";
}

def TokenSetOp : LPN_Op<"token.set", [Pure]> {
  let summary = "Append a `(key, value)` entry to a token log";

  let description = [{
    Append a new `(key, value)` tuple to the token's log and return the updated
    token SSA value.  If the same key appears multiple times, the most recent
    entry is the one observed by `token.get`.  This models the semantic history
    needed for later transition-merging passes without resorting to mutable
    dictionaries.
  }];

  let arguments = (ins
    TokenType:$token,
    StrAttr:$field,
    I64:$value
  );

  let results = (outs
    TokenType:$updated
  );

  let assemblyFormat =
    "$token `,` $field `,` $value attr-dict `:` type($token) `->` type($updated)";
}

def TokenGetOp : LPN_Op<"token.get", [Pure]> {
  let summary = "Read the latest value for a key from a token log";

  let description = [{
    Inspect the token's append-only log and return the value from the most
    recent entry whose key matches `field`.  If the key was never set, the
    runtime is allowed to return zero (mirroring the legacy simulator's
    behavior).
  }];

  let arguments = (ins
    TokenType:$token,
    StrAttr:$field
  );

  let results = (outs
    I64:$value
  );

  let assemblyFormat =
    "$token `,` $field attr-dict `:` type($token) `->` type($value)";
}

//===----------------------------------------------------------------------===//
// Control abstractions
//===----------------------------------------------------------------------===//

def ChoiceYieldOp : LPN_Op<"choice.yield", [Terminator]> {
  let summary = "Terminate a branch of lpn.choice";
  let assemblyFormat = "attr-dict";
}

def ChoiceOp : LPN_Op<"choice", [
    SingleBlockImplicitTerminator<"mlir::lpn::ChoiceYieldOp">
  ]> {
  let summary = "Abstract nondeterministic branch driven by hidden state";

  let description = [{
    Represents a branch whose predicate depends on state that has been dropped
    from the observable view of the net. Each region models one possible
    execution path; downstream passes may refine or discard the non-determinism
    once more information becomes available.
  }];

  let regions = (region AnyRegion:$thenRegion,
                 SizedRegion<1>:$elseRegion);

  let assemblyFormat = "attr-dict-with-keyword $thenRegion $elseRegion";
}

#endif // LPN_OPS_TD
