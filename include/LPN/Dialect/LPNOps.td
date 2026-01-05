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

def ArrayOp : LPN_Op<"array", [Pure]> {
  let summary = "Create a homogeneous SSA array";

  let description = [{
    Packs a variadic list of SSA values into a single `!lpn.array<elem>`
    container.  Every operand must have the same type, which becomes the array
    element type.
  }];

  let arguments = (ins
    Variadic<AnyType>:$elements
  );

  let results = (outs
    ArrayType:$array
  );

  let assemblyFormat =
    "$elements attr-dict `:` type($elements) `->` type($array)";
}

def ArrayGetOp : LPN_Op<"array.get", [Pure]> {
  let summary = "Read an element from an SSA array";

  let description = [{
    Load the element referenced by `index` from the given array.  The result
    type must match the array's element type.
  }];

  let arguments = (ins
    ArrayType:$array,
    Index:$index
  );

  let results = (outs
    AnyType:$element
  );

  let assemblyFormat =
    "$array `,` $index attr-dict `:` `(` type($array) `,` type($index) `)` `->` type($element)";
}

def ArraySetOp : LPN_Op<"array.set", [Pure]> {
  let summary = "Update an element in an SSA array";

  let description = [{
    Return a new array identical to `array` except that the element at `index`
    is replaced by `value`.
  }];

  let arguments = (ins
    ArrayType:$array,
    Index:$index,
    AnyType:$value
  );

  let results = (outs
    ArrayType:$result
  );

  let assemblyFormat =
    "$array `,` $index `,` $value attr-dict `:` `(` type($array) `,` type($index) `,` type($value) `)` `->` type($result)";
}

def ArrayLenOp : LPN_Op<"array.len", [Pure]> {
  let summary = "Get the length of an SSA array";

  let arguments = (ins
    ArrayType:$array
  );

  let results = (outs
    Index:$length
  );

  let assemblyFormat =
    "$array attr-dict `:` type($array) `->` type($length)";
}

def ArrayAllocOp : LPN_Op<"array.alloc", [Pure]> {
  let summary = "Allocate a new array of given size";
  let arguments = (ins Index:$size, AnyType:$fill_value);
  let results = (outs ArrayType:$result);
  let assemblyFormat = "$size `,` $fill_value attr-dict `:` type($size) `,` type($fill_value) `->` type($result)";
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

def TokenCreateOp : LPN_Op<"create", [Pure]> {
  let summary = "Create a token with an optional literal property log";

  let description = [{
    Allocate a fresh token handle of type `!lpn.token`.  Tokens are modeled as
    append-only logs of `(key, value)` pairs.  The optional `log_prefix`
    attribute allows the IR to materialize a compile-time log prefix (each
    dictionary entry is appended in attribute iteration order).  If omitted, the
    token starts with an empty log.
  }];

  let arguments = (ins
    Variadic<TokenType>:$deps,
    OptionalAttr<DictionaryAttr>:$log_prefix
  );

  let results = (outs
    TokenType:$token
  );

  let assemblyFormat =
    "($deps^ `:` type($deps))? ($log_prefix^)? attr-dict `:` type($token)";
}

def TokenCloneOp : LPN_Op<"clone", [Pure]> {
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

def KeyLiteralOp : LPN_Op<"key.literal", [Pure]> {
  let summary = "Materialize a log key from a literal string";

  let description = [{
    Produces a key handle that references the literal `value`.  Dynamic keys can
    be materialized from runtime register identifiers via `lpn.key.reg`.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs KeyType:$key);

  let assemblyFormat = "$value attr-dict `:` type($key)";
}

def KeyRegOp : LPN_Op<"key.reg", [Pure]> {
  let summary = "Wrap a runtime key register identifier as a metadata key";

  let description = [{
    Accepts an `i64` identifier (for example, the output of a hash/registry
    lookup) and produces a `!lpn.key` handle that can be consumed by
    `lpn.token.{get,set}`.  This enables fully dynamic metadata schemas without
    enumerating every key literal in the IR.
  }];

  let arguments = (ins I64:$id);
  let results = (outs KeyType:$key);

  let assemblyFormat =
    "$id attr-dict `:` type($id) `->` type($key)";
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
    KeyType:$key,
    I64:$value
  );

  let results = (outs
    TokenType:$updated
  );

  let assemblyFormat =
    "$token `,` $key `,` $value attr-dict `:` type($token) `,` type($key) `,` type($value) `->` type($updated)";
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
    KeyType:$key
  );

  let results = (outs
    I64:$value
  );

  let assemblyFormat =
    "$token `,` $key attr-dict `:` type($token) `,` type($key) `->` type($value)";
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
