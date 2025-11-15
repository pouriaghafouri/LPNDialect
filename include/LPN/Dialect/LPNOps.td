#ifndef LPN_OPS_TD
#define LPN_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

include "LPNDialect.td"
include "LPNTypes.td"
include "LPNAttributes.td"

class LPN_Op<string mnemonic, list<Trait> traits = []> : Op<LPN_Dialect, mnemonic, traits> {}

def NetOp : LPN_Op<"net", [IsolatedFromAbove]> {
  let summary = "Top-level container for a LPN";
  let description = [{
    `lpn.net` is a container operation that holds the whole LPN.
    Its single region typically contains `lpn.place`,
    `lpn.transition`, `lpn.in_edge` and `lpn.out_edge` operations
    describing the graph structure.
  }];

  let regions = (region AnyRegion:$body);

  // Simple syntax: `lpn.net { ... }`
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def PlaceOp : LPN_Op<"place", [NoSideEffect]> {
  let summary = "Create a place in a LPN";

  let description = [{
    Creates a place node and returns a handle of type `!lpn.node`
    that can be used when creating edges.

    - `name`           : human-readable identifier of the place.
    - `capacity`       : maximum number of tokens stored (optional, -1 == unbounded).
    - `initial_tokens` : number of tokens initially present (optional).
    - `token_props`    : schema of token properties for this place (optional).

    Example:
      %p = lpn.place "P_in"
             { capacity = 16, initial_tokens = 1,
               token_props = #lpn.token_props<["size", "prio"]> }
           : !lpn.node
  }];

  let arguments = (ins
    StrAttr:$name,

    // Optional integer capacity; if absent, interpreted as unbounded.
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<I64Attr>:$initial_tokens,
    OptionalAttr<TokenPropertySet>:$token_props
  );

  let results = (outs
    NodeType:$handle
  );

  let assemblyFormat = [{
    $name (`capacity` $capacity^)? (`,`
      `initial_tokens` $initial_tokens^)? (`,`
      `token_props` $token_props^)? attr-dict
      `:` type($handle)
  }];
}

def TransitionOp : LPN_Op<"transition", [NoSideEffect]> {
  let summary = "Create a transition in a LPN";

  let description = [{
    Creates a transition node and returns a handle of type `!lpn.node`.

    - `name`      : human-readable identifier of the transition.
    - `delay`     : constant latency (cycles) before firing (optional).
                   More advanced delay functions can later be modelled
                   by custom attributes referring to functions.
  }];

  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<I64Attr>:$delay
  );

  let results = (outs
    NodeType:$handle
  );

  let assemblyFormat = [{
    $name (`delay` $delay^)? attr-dict
      `:` type($handle)
  }];
}

// (place -> transition)

def InEdgeOp : LPN_Op<"in_edge", [NoSideEffect]> {
  let summary = "Input edge from place to transition";

  let description = [{
    Models an input edge from a place to a transition.

    - `src`       : handle of the source place.
    - `dst`       : handle of the destination transition.
    - `weight`    : number of tokens consumed when the transition commits.
    - `guard`     : (optional) textual guard condition on token properties.
    - `threshold` : (optional) threshold used in the enable condition.

    In the original Python library, weights/guards/thresholds are
    functions over tokens; here we start with constants and simple
    strings to keep the first dialect version easy to work with.
  }];

  let arguments = (ins
    NodeType:$src,
    NodeType:$dst,

    OptionalAttr<I64Attr>:$weight,
    OptionalAttr<StrAttr>:$guard,
    OptionalAttr<I64Attr>:$threshold
  );

  let results = (outs);

  let assemblyFormat = [{
    $src `->` $dst
      (`weight` $weight^)? (`,`
       `guard` $guard^)? (`,`
       `threshold` $threshold^)? attr-dict
      `:` type($src) `,` type($dst)
  }];
}

// (transition -> place)

def OutEdgeOp : LPN_Op<"out_edge", [NoSideEffect]> {
  let summary = "Output edge from transition to place";

  let description = [{
    Models an output edge from a transition to a place.

    - `src`    : handle of the source transition.
    - `dst`    : handle of the destination place.
    - `weight` : number of tokens produced when the transition commits.
  }];

  let arguments = (ins
    NodeType:$src,
    NodeType:$dst,
    OptionalAttr<I64Attr>:$weight
  );

  let results = (outs);

  let assemblyFormat = [{
    $src `->` $dst
      (`weight` $weight^)? attr-dict
      `:` type($src) `,` type($dst)
  }];
}

#endif // LPN_OPS_TD
