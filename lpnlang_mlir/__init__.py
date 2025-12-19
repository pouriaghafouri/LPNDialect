from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union


class Statement:
  def render(self, indent: str) -> str:
    raise NotImplementedError


class RawStatement(Statement):
  def __init__(self, text: str):
    self.text = text

  def render(self, indent: str) -> str:
    return f"{indent}{self.text}"


class IfStatement(Statement):
  def __init__(self,
               cond: str,
               then_stmts: List[Statement],
               else_stmts: Optional[List[Statement]] = None):
    self.cond = cond
    self.then_stmts = then_stmts
    self.else_stmts = else_stmts

  def _render_block(self, stmts: List[Statement], indent: str) -> List[str]:
    rendered = []
    inner = indent + "  "
    for stmt in stmts:
      rendered.append(stmt.render(inner))
    rendered.append(f"{inner}scf.yield")
    return rendered

  def render(self, indent: str) -> str:
    lines = [f"{indent}scf.if {self.cond} {{"]
    lines.extend(self._render_block(self.then_stmts, indent))
    lines.append(f"{indent}}}")
    if self.else_stmts is not None:
      lines.append(f"{indent}else {{")
      lines.extend(self._render_block(self.else_stmts, indent))
      lines.append(f"{indent}}}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Value:
  builder: "TransitionBuilder"
  name: str
  typ: str

  def __str__(self) -> str:
    return self.name

  def __bool__(self) -> bool:
    raise TypeError("LPN SSA values cannot be used as Python booleans")

  def _require_numeric(self) -> None:
    if self.typ not in ("i64", "f64"):
      raise TypeError(f"operation not supported on values of type {self.typ}")

  def __add__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.addf(self, other)
    return self.builder.addi(self, other, typ=self.typ)

  def __sub__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.subf(self, other)
    return self.builder.subi(self, other, typ=self.typ)

  def __truediv__(self, other: Union["Value", int, float]) -> "Value":
    if self.typ != "f64":
      raise TypeError("division only supported for f64 values")
    return self.builder.divf(self, other)

  def eq(self, other: Union["Value", int, float]) -> "Value":
    if self.typ not in ("i64", "index"):
      raise TypeError("eq is only defined on integer/index values")
    return self.builder.cmpi("eq", self, other, typ=self.typ)

  def ne(self, other: Union["Value", int, float]) -> "Value":
    if self.typ not in ("i64", "index"):
      raise TypeError("ne is only defined on integer/index values")
    return self.builder.cmpi("ne", self, other, typ=self.typ)


@dataclass(frozen=True)
class TokenValue:
  builder: "TransitionBuilder"
  name: str

  def __str__(self) -> str:
    return self.name

  def get(self, field: str) -> Value:
    return self.builder.token_get(self, field)

  def set(self, field: str, value: Union[Value, int]) -> "TokenValue":
    return self.builder.token_set(self, field, value)

  def clone(self) -> "TokenValue":
    return self.builder.token_clone(self)


@dataclass(frozen=True)
class PlaceHandle:
  name: str


class TransitionBuilder:
  _global_id = 0

  def __init__(self, name: str):
    self.name = name
    self._ops: List[Statement] = []
    self._value_id = 0
    self._place_handles: dict[str, Value] = {}
    self._local_prefix = TransitionBuilder._global_id
    TransitionBuilder._global_id += 1

  def _next_value(self) -> str:
    name = f"%t{self._local_prefix}_{self._value_id}"
    self._value_id += 1
    return name

  def _append(self, text: str) -> None:
    self._ops.append(RawStatement(text))

  def _wrap_value(self, name: str, typ: str) -> Value:
    return Value(self, name, typ)

  def _wrap_token(self, name: str) -> TokenValue:
    return TokenValue(self, name)

  def _coerce_value(self, value: Union[Value, int, float], typ: str) -> Value:
    if isinstance(value, Value):
      if value.typ != typ:
        raise TypeError(f"expected value of type {typ}, but got {value.typ}")
      return value
    if typ == "i64":
      return self.const_i64(int(value))
    if typ == "f64":
      return self.const_f64(float(value))
    if typ == "index":
      if isinstance(value, int):
        return self.const_index(value)
    raise TypeError(f"cannot coerce value of type {type(value)} to {typ}")

  def _resolve_place_operand(self,
                             place: Union[PlaceHandle, Value]) -> Value:
    if isinstance(place, PlaceHandle):
      return self._ensure_place_handle(place)
    if isinstance(place, Value) and place.typ == "!lpn.place":
      return place
    raise TypeError("expected a place handle or SSA place reference")

  def _ensure_place_handle(self, place: PlaceHandle) -> Value:
    handle = self._place_handles.get(place.name)
    if handle is None:
      name = self._next_value()
      self._ops.append(
          RawStatement(f"{name} = lpn.place_ref @{place.name} : !lpn.place"))
      handle = self._wrap_value(name, "!lpn.place")
      self._place_handles[place.name] = handle
    return handle

  def const_f64(self, value: float) -> Value:
    literal = f"{float(value):.6f}"
    name = self._next_value()
    self._append(f"{name} = arith.constant {literal} : f64")
    return self._wrap_value(name, "f64")

  def f64(self, value: float) -> Value:
    return self.const_f64(value)

  def const_i64(self, value: int) -> Value:
    name = self._next_value()
    self._append(f"{name} = arith.constant {int(value)} : i64")
    return self._wrap_value(name, "i64")

  def i64(self, value: int) -> Value:
    return self.const_i64(value)

  def const_index(self, value: int) -> Value:
    name = self._next_value()
    self._append(f"{name} = arith.constant {int(value)} : index")
    return self._wrap_value(name, "index")

  def index(self, value: int) -> Value:
    return self.const_index(value)

  def addi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.addi {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def subi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.subi {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def addf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.addf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def subf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.subf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def divf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.divf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def sitofp(self,
             value: Union[Value, int],
             *,
             src_type: str = "i64",
             dst_type: str = "f64") -> Value:
    src_val = self._coerce_value(value, src_type)
    name = self._next_value()
    self._append(f"{name} = arith.sitofp {src_val} : {src_type} to {dst_type}")
    return self._wrap_value(name, dst_type)

  def index_cast(self,
                 value: Union[Value, int],
                 *,
                 src_type: str,
                 dst_type: str = "index") -> Value:
    src_val = self._coerce_value(value, src_type)
    name = self._next_value()
    self._append(
        f"{name} = arith.index_cast {src_val} : {src_type} to {dst_type}")
    return self._wrap_value(name, dst_type)

  def cmpi(self,
           predicate: str,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.cmpi {predicate}, {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, "i1")

  def _capture_ops(
      self, fn: Optional[Callable[['TransitionBuilder'], None]]
  ) -> List[Statement]:
    if fn is None:
      return []
    saved_ops = self._ops
    branch_ops: List[Statement] = []
    self._ops = branch_ops
    try:
      fn(self)
    finally:
      self._ops = saved_ops
    return branch_ops

  def if_op(
      self,
      cond: Union[Value, str],
      true_fn: Callable[['TransitionBuilder'], None],
      false_fn: Optional[Callable[['TransitionBuilder'], None]] = None) -> None:
    cond_name = cond.name if isinstance(cond, Value) else cond
    true_ops = self._capture_ops(true_fn)
    false_ops = self._capture_ops(false_fn) if false_fn else None
    self._ops.append(IfStatement(cond_name, true_ops, false_ops))

  def _ensure_delay(self, delay: Optional[Union[Value, float, int]]) -> Value:
    if delay is None:
      return self.const_f64(0.0)
    if isinstance(delay, Value):
      if delay.typ != "f64":
        raise TypeError("delay must be an f64 value")
      return delay
    return self.const_f64(float(delay))

  def take(self, place: Union[PlaceHandle, Value]) -> TokenValue:
    handle = self._resolve_place_operand(place)
    name = self._next_value()
    self._append(
        f"{name} = lpn.take {handle} : !lpn.place -> !lpn.token")
    return self._wrap_token(name)

  def take_handle(self, handle: Value) -> TokenValue:
    if handle.typ != "!lpn.place":
      raise TypeError("take_handle expects a !lpn.place value")
    name = self._next_value()
    self._append(
        f"{name} = lpn.take {handle} : !lpn.place -> !lpn.token")
    return self._wrap_token(name)

  def emit(self,
           place: Union[PlaceHandle, Value],
           token: TokenValue,
           delay: Optional[Union[Value, float, int]] = None) -> None:
    handle = self._resolve_place_operand(place)
    delay_value = self._ensure_delay(delay)
    self._append(
        f"lpn.emit {handle}, {token.name}, {delay_value} : !lpn.place, !lpn.token, f64")

  def emit_handle(self,
                  handle: Value,
                  token: TokenValue,
                  delay: Optional[Union[Value, float, int]] = None) -> None:
    if handle.typ != "!lpn.place":
      raise TypeError("emit_handle expects a !lpn.place SSA value")
    delay_value = self._ensure_delay(delay)
    self._append(
        f"lpn.emit {handle}, {token.name}, {delay_value} : !lpn.place, !lpn.token, f64")

  def count(self, place: Union[PlaceHandle, Value]) -> Value:
    handle = self._resolve_place_operand(place)
    name = self._next_value()
    self._append(f"{name} = lpn.count {handle} : !lpn.place -> i64")
    return self._wrap_value(name, "i64")

  def token_get(self, token: TokenValue, field: str) -> Value:
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.get\"({token.name}) {{field = \"{field}\"}} : (!lpn.token) -> i64")
    return self._wrap_value(name, "i64")

  def token_set(self,
                token: TokenValue,
                field: str,
                value_ssa: Union[Value, int]) -> TokenValue:
    value = self._coerce_value(value_ssa, "i64")
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.set\"({token.name}, {value}) {{field = \"{field}\"}} : (!lpn.token, i64) -> !lpn.token")
    return self._wrap_token(name)

  def token_clone(self, token: TokenValue) -> TokenValue:
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.clone\"({token.name}) : (!lpn.token) -> !lpn.token")
    return self._wrap_token(name)

  def token_create(self, properties: Optional[Dict[str, int]] = None) -> TokenValue:
    props_dict = properties or {}
    props = ", ".join(
        f"{key} = {value} : i64" for key, value in sorted(props_dict.items()))
    attr = f"{{{props}}}" if props else "{}"
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.create\"() {{log_prefix = {attr}}} : () -> !lpn.token")
    return self._wrap_token(name)

  def place_list(self, places: Sequence[PlaceHandle]) -> Value:
    if not places:
      raise ValueError("place_list requires at least one place")
    refs = ", ".join(f"@{place.name}" for place in places)
    name = self._next_value()
    self._append(
        f"{name} = lpn.place_list {{places = [{refs}]}} : !lpn.place_list")
    return self._wrap_value(name, "!lpn.place_list")

  def place_list_get(self,
                     place_list: Value,
                     index: Union[Value, int]) -> Value:
    if place_list.typ != "!lpn.place_list":
      raise TypeError("place_list_get expects a !lpn.place_list value")
    idx_value = self._coerce_value(index, "index")
    name = self._next_value()
    self._append(
        f"{name} = lpn.place_list.get {place_list}, {idx_value} : (!lpn.place_list, index) -> !lpn.place")
    return self._wrap_value(name, "!lpn.place")

  def materialize_place(self, place: PlaceHandle) -> None:
    self._ensure_place_handle(place)

  def render(self) -> List[str]:
    ops = []
    if self._ops:
      for op in self._ops:
        ops.append(op.render("        "))
    ops.append("        lpn.schedule.return")
    return ops


class NetBuilder:
  """Tiny DSL for emitting the new MLIR dialect."""

  def __init__(self, name: str = "net"):
    self.name = name
    self._places: List[tuple[PlaceHandle, Optional[int], Optional[int], bool]] = []
    self._transitions: List[TransitionBuilder] = []

  def place(self,
            name: str,
            *,
            capacity: Optional[int] = None,
            initial_tokens: Optional[int] = None,
            observable: bool = False) -> PlaceHandle:
    handle = PlaceHandle(name)
    self._places.append((handle, capacity, initial_tokens, observable))
    return handle

  def transition(self, name: str) -> Callable[[Callable[[TransitionBuilder], None]], Callable[[TransitionBuilder], None]]:
    def decorator(fn: Callable[[TransitionBuilder], None]):
      builder = TransitionBuilder(name)
      fn(builder)
      self._transitions.append(builder)
      return fn
    return decorator

  def build(self) -> str:
    lines = ["module {", "  lpn.net {"]

    for place, capacity, initial, observable in self._places:
      attrs = []
      if capacity is not None:
        attrs.append(f"capacity = {capacity} : i64")
      if initial is not None:
        attrs.append(f"initial_tokens = {initial} : i64")
      if observable:
        attrs.append("observable")
      attr_text = ""
      if attrs:
        attr_text = " {" + ", ".join(attrs) + "}"
      lines.append(f"    lpn.place @{place.name}{attr_text}")

    for transition in self._transitions:
      lines.append(f"    lpn.transition @{transition.name} {{")
      lines.append("      ^bb0:")
      lines.extend(transition.render())
      lines.append("    }")

    lines.append("    lpn.halt")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)

  def emit_to_file(self, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(self.build())
