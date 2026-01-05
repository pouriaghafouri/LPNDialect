from __future__ import annotations

import ast
import hashlib
import heapq
import inspect
import itertools
import operator
import re
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, get_args, get_origin


_AST_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.MatMult: operator.matmul,
}


_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _to_signed_i64(value: int) -> int:
  mask = (1 << 64) - 1
  value &= mask
  if value >= (1 << 63):
    value -= (1 << 64)
  return value


_FNV_OFFSET_BASIS_I64 = _to_signed_i64(_FNV_OFFSET_BASIS)


def _hash_literal_segment_to_i64(text: str) -> int:
  if not text:
    return 0
  digest = hashlib.sha256(text.encode("utf-8")).digest()
  raw = int.from_bytes(digest[:8], byteorder="little", signed=False)
  return _to_signed_i64(raw)


_MISSING = object()


class _GlobalNameInjector:
  def __init__(self, target: Dict[str, Any], mapping: Dict[str, Any]):
    self._target = target
    self._mapping = mapping
    self._previous: Dict[str, Any] = {}

  def __enter__(self) -> None:
    for name, value in self._mapping.items():
      if self._target.get(name) is value:
        continue
      previous = self._target.get(name, _MISSING)
      self._previous[name] = previous
      self._target[name] = value

  def __exit__(self, exc_type, exc, tb) -> None:
    for name, previous in self._previous.items():
      if previous is _MISSING:
        self._target.pop(name, None)
      else:
        self._target[name] = previous
    self._previous.clear()


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


class ForStatement(Statement):
  def __init__(self, iv: str, lower: str, upper: str, step: str,
               body: List[Statement],
               iter_args: Sequence[Tuple[str, str, str]] = (),
               results: Sequence[str] = (),
               yield_values: Sequence[str] = (),
               yield_types: Sequence[str] = ()):
    self.iv = iv
    self.lower = lower
    self.upper = upper
    self.step = step
    self.body = body
    self.iter_args = iter_args
    self.results = results
    self.yield_values = yield_values
    self.yield_types = yield_types

  def render(self, indent: str) -> str:
    iter_part = ""
    result_part = ""
    if self.iter_args:
      args_def = ", ".join(f"{name} = {init}" for name, init, _ in self.iter_args)
      types = ", ".join(typ for _, _, typ in self.iter_args)
      iter_part = f" iter_args({args_def}) -> ({types})"

      if self.results:
        res_names = ", ".join(self.results)
        result_part = f"{res_names} = "

    lines = [f"{indent}{result_part}scf.for {self.iv} = {self.lower} to {self.upper} step {self.step}{iter_part} {{"]
    inner = indent + "  "
    for stmt in self.body:
      lines.append(stmt.render(inner))

    if self.yield_values:
      vals = ", ".join(self.yield_values)
      typs = ", ".join(self.yield_types)
      lines.append(f"{inner}scf.yield {vals} : {typs}")
    else:
      lines.append(f"{inner}scf.yield")

    lines.append(f"{indent}}}")
    return "\n".join(lines)


class TransitionScriptExecutor:
  """Very small AST executor that rewrites Python if-statements to scf.if."""

  def __init__(self,
               fn: Callable[..., None],
               *,
               require_builder_arg: bool = True):
    self.fn = fn
    self.require_builder_arg = require_builder_arg
    self.filename = inspect.getsourcefile(fn) or fn.__code__.co_filename
    if not self.filename:
      raise ValueError("unable to locate source file for transition script")
    try:
      with open(self.filename, "r", encoding="utf-8") as f:
        file_source = f.read()
    except OSError as exc:
      raise ValueError("unable to read transition source") from exc

    module = ast.parse(file_source, filename=self.filename)
    first_line = fn.__code__.co_firstlineno

    def matches_function(node: ast.FunctionDef) -> bool:
      deco_lines = len(getattr(node, "decorator_list", []))
      start_line = node.lineno - deco_lines if deco_lines else node.lineno
      return start_line <= first_line <= node.lineno

    func_defs = [
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
        and matches_function(node)
    ]
    if not func_defs:
      func_defs = [
          node for node in ast.walk(module)
          if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
      ]
    if not func_defs:
      raise ValueError("transition script could not find the function body")
    self.func_def = func_defs[0]
    self.signature = inspect.signature(fn)

  def __call__(self, builder: "TransitionBuilder", *args: Any,
               **kwargs: Any) -> None:
    if self.require_builder_arg:
      try:
        bound = self.signature.bind_partial(builder, *args, **kwargs)
      except TypeError as exc:
        raise TypeError(
            f"transition script '{self.fn.__name__}' must accept a builder argument"
        ) from exc
    else:
      # params = list(self.signature.parameters.values())
      # if params:
      #   first = params[0]
      #   if (first.kind in (inspect.Parameter.POSITIONAL_ONLY,
      #                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
      #       and first.default is inspect._empty):
      #     raise TypeError(
      #         f"jit transition '{self.fn.__name__}' should not declare an explicit builder argument"
      #     )
      bound = self.signature.bind_partial(*args, **kwargs)

    bound.apply_defaults()
    self.builder = builder
    self.locals: Dict[str, Any] = dict(bound.arguments)
    self.locals.setdefault("builder", builder)
    closure_vars = inspect.getclosurevars(self.fn)
    self.locals.update(closure_vars.nonlocals)
    builder_globals = self._collect_builder_globals(builder)
    for name, value in builder_globals.items():
      if name == "builder":
        continue
      self.locals.setdefault(name, value)
    self.globals: Dict[str, Any] = dict(self.fn.__globals__)
    self.globals.setdefault("__builtins__", __builtins__)
    self.globals.update(builder_globals)

    module_globals = self.fn.__globals__
    injector = _GlobalNameInjector(module_globals, builder_globals)
    injector.__enter__()
    try:
      self._exec_block(self.func_def.body)
    finally:
      injector.__exit__(None, None, None)

  def _exec_block(self, statements: List[ast.stmt]) -> None:
    for stmt in statements:
      self._exec_stmt(stmt)

  def _collect_builder_globals(
      self, builder: "TransitionBuilder") -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for attr in dir(builder):
      if attr.startswith("_"):
        continue
      value = getattr(builder, attr)
      if callable(value):
        mapping[attr] = value
    mapping.setdefault("builder", builder)
    return mapping

  def _exec_stmt(self, stmt: ast.stmt) -> None:
    if isinstance(stmt, ast.If):
      self._exec_if(stmt)
      return
    if isinstance(stmt, ast.For):
      if self._is_range_loop(stmt):
        self._exec_range_loop(stmt)
        return
      if self._try_unroll_loop(stmt):
        return
      raise TypeError(
          "unsupported for-loop: transitions must use range(...) or an iterable known at build time")
    if isinstance(stmt, ast.While):
      self._exec_while(stmt)
      return
    if isinstance(stmt, ast.Assign) and self._try_exec_assign(stmt):
      return
    if isinstance(stmt, ast.AugAssign) and self._try_exec_aug_assign(stmt):
      return

    mod = ast.Module(body=[stmt], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, self.filename, "exec")
    exec(code, self.globals, self.locals)

  def _try_unroll_loop(self, node: ast.For) -> bool:
    try:
      iterable = self._eval_expr(node.iter)
    except Exception:
      return False
    if isinstance(iterable, Value):
      return False
    try:
      iterator = iter(iterable)
    except TypeError:
      return False

    iteration = 0
    saved_locals = self.locals
    for item in iterator:
      iteration += 1
      if iteration > 10_000_000:
        raise RuntimeError("loop unrolling exceeded 10M iterations")
      loop_locals = dict(saved_locals)
      target = node.target
      if isinstance(target, ast.Name):
        loop_locals[target.id] = item
      elif isinstance(target, (ast.Tuple, ast.List)):
        try:
          values = list(item)
        except TypeError:
          return False
        if len(values) != len(target.elts):
          raise ValueError("loop tuple destructuring mismatch")
        for elt, value in zip(target.elts, values):
          if not isinstance(elt, ast.Name):
            raise TypeError("only simple tuple targets are supported in jit loops")
          loop_locals[elt.id] = value
      else:
        raise TypeError("unsupported loop target in jit transition")
      self.locals = loop_locals
      try:
        self._exec_block(node.body)
      finally:
        self.locals = saved_locals
    return True

  def _try_exec_assign(self, stmt: ast.Assign) -> bool:
    if len(stmt.targets) != 1:
      return False
    target = stmt.targets[0]
    if not isinstance(target, ast.Subscript):
      return False
    if not isinstance(target.value, ast.Name):
      return False

    var_name = target.value.id
    if var_name not in self.locals:
      return False

    obj = self.locals[var_name]
    if not isinstance(obj, Value):
      return False

    idx_node = target.slice
    if sys.version_info < (3, 9) and isinstance(idx_node, ast.Index):
      idx_node = idx_node.value

    try:
      index = self._eval_expr(idx_node)
      value = self._eval_expr(stmt.value)
    except Exception:
      return False

    new_obj = obj.set(index, value)
    self.locals[var_name] = new_obj
    return True

  def _try_exec_aug_assign(self, stmt: ast.AugAssign) -> bool:
    target = stmt.target
    if not isinstance(target, ast.Subscript):
      return False
    if not isinstance(target.value, ast.Name):
      return False

    var_name = target.value.id
    if var_name not in self.locals:
      return False

    obj = self.locals[var_name]
    if not isinstance(obj, Value):
      return False

    idx_node = target.slice
    if sys.version_info < (3, 9) and isinstance(idx_node, ast.Index):
      idx_node = idx_node.value

    op_type = type(stmt.op)
    op_func = _AST_OP_MAP.get(op_type)
    if not op_func:
      return False

    try:
      index = self._eval_expr(idx_node)
      operand = self._eval_expr(stmt.value)
    except Exception:
      return False

    current_val = obj[index]
    new_val = op_func(current_val, operand)
    new_obj = obj.set(index, new_val)
    self.locals[var_name] = new_obj
    return True

  def _eval_expr(self, expr: ast.expr) -> Any:
    node = ast.Expression(expr)
    ast.fix_missing_locations(node)
    code = compile(node, self.filename, "eval")
    return eval(code, self.globals, self.locals)

  def _exec_branch(self, statements: List[ast.stmt]) -> None:
    saved_locals = self.locals
    branch_locals = dict(saved_locals)
    self.locals = branch_locals
    try:
      self._exec_block(statements)
    finally:
      self.locals = saved_locals

  def _exec_if(self, node: ast.If) -> None:
    cond_value = self._eval_expr(node.test)
    if not isinstance(cond_value, Value):
      raise TypeError("transition conditions must evaluate to LPN SSA values")

    def then_fn(_builder: "TransitionBuilder") -> None:
      self._exec_branch(node.body)

    def else_fn(_builder: "TransitionBuilder") -> None:
      self._exec_branch(node.orelse)

    false_fn = else_fn if node.orelse else None
    self.builder.if_op(cond_value, then_fn, false_fn)

  def _exec_while(self, node: ast.While) -> None:
    if node.orelse:
      raise TypeError("while-loops in jit transitions do not support 'else'")
    iteration = 0
    while True:
      cond_value = self._eval_expr(node.test)
      if isinstance(cond_value, Value):
        raise TypeError("while-loop conditions must evaluate to Python scalars")
      if not bool(cond_value):
        break
      iteration += 1
      if iteration > 10_000_000:
        raise RuntimeError("while-loop exceeded 10M iterations while building the net")
      self._exec_block(node.body)

  def _is_range_loop(self, node: ast.For) -> bool:
    call = node.iter
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"

  def _exec_range_loop(self, node: ast.For) -> None:
    if node.orelse:
      raise TypeError("lpn jit for-loops do not support 'else' blocks")
    if not isinstance(node.target, ast.Name):
      raise TypeError("for-loop target must be a simple variable name")
    call = node.iter
    if call.keywords:
      raise TypeError("range(...) with keyword arguments is not supported")
    arg_count = len(call.args)
    if arg_count == 1:
      lower = 0
      upper = self._eval_expr(call.args[0])
      step = 1
    elif arg_count == 2:
      lower = self._eval_expr(call.args[0])
      upper = self._eval_expr(call.args[1])
      step = 1
    elif arg_count == 3:
      lower = self._eval_expr(call.args[0])
      upper = self._eval_expr(call.args[1])
      step = self._eval_expr(call.args[2])
      if isinstance(step, int) and step == 0:
        raise ValueError("range() step cannot be zero in jit loops")
    else:
      raise TypeError("range() expects 1-3 positional arguments in jit loops")

    # Find assignments in the loop body
    finder = AssignmentFinder()
    finder.visit(node)
    assigned_vars = sorted(list(finder.assigned))

    iter_args = []
    iter_arg_names = []
    for name in assigned_vars:
      if name in self.locals and isinstance(self.locals[name], Value):
        iter_args.append(self.locals[name])
        iter_arg_names.append(name)

    def loop_body(_builder: "TransitionBuilder", iv_value: Value, *args: Value) -> Sequence[Value]:
      saved_locals = self.locals
      loop_locals = dict(saved_locals)
      loop_locals[node.target.id] = iv_value
      for name, val in zip(iter_arg_names, args):
        loop_locals[name] = val
      self.locals = loop_locals
      try:
        self._exec_block(node.body)
        results = []
        for name in iter_arg_names:
          results.append(self.locals[name])
        return results
      finally:
        self.locals = saved_locals

    results = self.builder.for_range(lower, upper, step=step, iter_args=iter_args, body=loop_body)

    if iter_args:
      if not isinstance(results, tuple):
        results = (results,)
      for name, val in zip(iter_arg_names, results):
        self.locals[name] = val


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
    if self.typ not in ("i64", "f64", "index"):
      raise TypeError(f"operation not supported on values of type {self.typ}")

  def _require_integer_like(self) -> None:
    if self.typ not in ("i64", "index"):
      raise TypeError("comparison is only defined on integer/index values")

  def _cmp_int(self, predicate: str, other: Union["Value", int]) -> "Value":
    self._require_integer_like()
    return self.builder.cmpi(predicate, self, other, typ=self.typ)

  def __add__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.addf(self, other)
    return self.builder.addi(self, other, typ=self.typ)

  def __radd__(self, other: Union["Value", int, float]) -> "Value":
    return self.__add__(other)

  def __sub__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.subf(self, other)
    return self.builder.subi(self, other, typ=self.typ)

  def __rsub__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.subf(other, self)
    return self.builder.subi(other, self, typ=self.typ)

  def __truediv__(self, other: Union["Value", int, float]) -> "Value":
    if self.typ != "f64":
      raise TypeError("division only supported for f64 values")
    return self.builder.divf(self, other)

  def __mod__(self, other: Union["Value", int]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.remf(self, other)
    return self.builder.remi(self, other, typ=self.typ)

  def __rmod__(self, other: Union["Value", int]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.remf(other, self)
    return self.builder.remi(other, self, typ=self.typ)

  def __eq__(self, other: object) -> "Value":  # type: ignore[override]
    if isinstance(other, (Value, int)):
      return self._cmp_int("eq", other)
    return NotImplemented

  def __ne__(self, other: object) -> "Value":  # type: ignore[override]
    if isinstance(other, (Value, int)):
      return self._cmp_int("ne", other)
    return NotImplemented

  def __lt__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("slt", other)

  def __le__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sle", other)

  def __gt__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sgt", other)

  def __ge__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sge", other)

  def __getitem__(self, index: Union["Value", int]) -> "Value":
    return self.builder.array_get(self, index)

  def __setitem__(self, index: Union["Value", int], value: Union["Value", int, float]) -> None:
    raise TypeError("LPN arrays are immutable SSA values; use 'new_arr = arr.set(index, value)' instead of 'arr[index] = value'.")

  def set(self, index: Union["Value", int, str, "KeyValue"], value: Union["Value", int, float]) -> "Value":
    if self.typ == "!lpn.token":
      return self.builder.token_set(TokenValue(self.builder, self.name), index, value)
    return self.builder.array_set(self, index, value)

  def get(self, key: Union[str, "KeyValue", Value, int]) -> Value:
    if self.typ == "!lpn.token":
      return self.builder.token_get(TokenValue(self.builder, self.name), key)
    raise TypeError(f"get() not supported on values of type {self.typ}")

  def clone(self) -> "TokenValue":
    if self.typ == "!lpn.token":
      return self.builder.clone(TokenValue(self.builder, self.name))
    raise TypeError(f"clone() not supported on values of type {self.typ}")

  def len(self) -> "Value":
    return self.builder.array_len(self)

  def __format__(self, format_spec: str) -> str:
    if format_spec not in ("", "s"):
      raise ValueError(
          f"LPN SSA values only support empty format specifiers, got '{format_spec}'"
      )
    return self.builder._register_format_placeholder(self)

  def eq(self, other: Union["Value", int, float]) -> "Value":
    return self.__eq__(other)

  def ne(self, other: Union["Value", int, float]) -> "Value":
    return self.__ne__(other)


@dataclass(frozen=True)
class TokenValue:
  builder: "TransitionBuilder"
  name: str

  def __str__(self) -> str:
    return self.name

  def get(self, key: Union[str, "KeyValue", Value, int]) -> Value:
    return self.builder.token_get(self, key)

  def set(self,
          key: Union[str, "KeyValue", Value, int],
          value: Union[Value, int]) -> "TokenValue":
    return self.builder.token_set(self, key, value)

  def clone(self) -> "TokenValue":
    return self.builder.clone(self)


@dataclass(frozen=True)
class KeyValue:
  builder: "TransitionBuilder"
  name: str

  def __str__(self) -> str:
    return self.name

  def as_value(self) -> Value:
    return self.builder._ssa_values[self.name]


@dataclass(frozen=True)
class PlaceHandle:
  name: str


class TransitionBuilder:
  _global_id = 0
  _value_name_re = re.compile(r"%[A-Za-z0-9_$.]+")

  def __init__(self, name: str, owner: Optional["NetBuilder"] = None):
    self.name = name
    self._owner = owner
    self._ops: List[Statement] = []
    self._value_id = 0
    self._place_handles: dict[str, Value] = {}
    self._literal_keys: dict[str, KeyValue] = {}
    self._ssa_values: dict[str, Value] = {}
    self._local_prefix = TransitionBuilder._global_id
    TransitionBuilder._global_id += 1
    self._literal_segment_hashes: Dict[str, int] = {}
    self._literal_segment_values: Dict[str, Value] = {}

  def _next_value(self) -> str:
    name = f"%t{self._local_prefix}_{self._value_id}"
    self._value_id += 1
    return name

  def _register_format_placeholder(self, value: Value) -> str:
    return value.name

  def _append(self, text: str) -> None:
    self._ops.append(RawStatement(text))

  def _wrap_value(self, name: str, typ: str) -> Value:
    value = Value(self, name, typ)
    self._ssa_values[name] = value
    return value

  def _get_or_create_value(self, name: str, typ: str) -> Value:
    existing = self._ssa_values.get(name)
    if existing is not None:
      if existing.typ != typ:
        raise TypeError(
            f"SSA value '%{name}' already registered with type {existing.typ}, expected {typ}")
      return existing
    value = Value(self, name, typ)
    self._ssa_values[name] = value
    return value

  def _wrap_token(self, name: str) -> TokenValue:
    return TokenValue(self, name)

  def _wrap_key(self, name: str) -> KeyValue:
    self._wrap_value(name, "!lpn.key")
    return KeyValue(self, name)

  def _register_argument(self, name: str, typ: str) -> Value:
    value = Value(self, name, typ)
    self._ssa_values[name] = value
    return value

  def _coerce_value(self, value: Union[Value, int, float], typ: str) -> Value:
    if isinstance(value, Value):
      if value.typ == typ:
        return value
      if value.typ == "index" and typ == "i64":
        return self.index_cast(value, src_type="index", dst_type="i64")
      if value.typ == "i64" and typ == "index":
        return self.index_cast(value, src_type="i64", dst_type="index")
      raise TypeError(f"expected value of type {typ}, but got {value.typ}")
    if typ == "i64":
      return self.const_i64(int(value))
    if typ == "f64":
      return self.const_f64(float(value))
    if typ == "index":
      if isinstance(value, int):
        return self.const_index(value)
    raise TypeError(f"cannot coerce value of type {type(value)} to {typ}")

  def _as_value(self, value: Union[Value, PlaceHandle, int, float]) -> Value:
    if isinstance(value, Value):
      return value
    if isinstance(value, KeyValue):
      return value.as_value()
    if isinstance(value, TokenValue):
      return self._ensure_token_value(value)
    if isinstance(value, PlaceHandle):
      return self._ensure_place_handle(value)
    if isinstance(value, float):
      return self.const_f64(value)
    if isinstance(value, int):
      return self.const_i64(value)
    raise TypeError(f"unsupported element type {type(value)}")

  def _ensure_token_value(self, token: TokenValue) -> Value:
    if token.builder is not self:
      raise ValueError("token value belongs to a different builder")
    return self._get_or_create_value(token.name, "!lpn.token")

  def _coerce_function_argument(self, value: Any, typ: str) -> Value:
    if typ == "!lpn.token":
      if isinstance(value, TokenValue):
        return self._ensure_token_value(value)
      if isinstance(value, Value) and value.typ == "!lpn.token":
        return value
      raise TypeError("expected a TokenValue or !lpn.token SSA value")
    if typ == "!lpn.place":
      if isinstance(value, Value) and value.typ == "!lpn.place":
        return value
      return self._resolve_place_operand(value)
    if typ == "!lpn.key":
      if isinstance(value, KeyValue):
        return value.as_value()
      if isinstance(value, Value) and value.typ == "!lpn.key":
        return value
      key = self._ensure_key(value)
      return key.as_value()
    if isinstance(value, Value):
      if value.typ != typ:
        raise TypeError(f"expected value of type {typ}, but got {value.typ}")
      return value
    return self._coerce_value(value, typ)

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

  def _array_element_type(self, array_type: str) -> str:
    prefix = "!lpn.array<"
    if array_type.startswith(prefix) and array_type.endswith(">"):
      return array_type[len(prefix):-1]
    raise TypeError("expected an !lpn.array value")

  def _coerce_array_element(self,
                            element: Union[Value, KeyValue, PlaceHandle, int, float],
                            expected_type: Optional[str] = None) -> Value:
    if isinstance(element, (Value, KeyValue, PlaceHandle)):
      value = self._as_value(element)
    elif isinstance(element, float):
      target = expected_type if expected_type == "f64" else "f64"
      value = self._coerce_value(element, target)
    elif isinstance(element, int):
      target = expected_type if expected_type in ("i64", "index") else "i64"
      if target == "index":
        value = self.const_index(element)
      else:
        value = self.const_i64(element)
    else:
      value = self._as_value(element)
    if expected_type is not None and value.typ != expected_type:
      raise TypeError(
          f"array elements must all have type {expected_type}, saw {value.typ}")
    return value

  def array(self,
            *elements: Union[Value, PlaceHandle, int, float,
                             Sequence[Union[Value, PlaceHandle, int, float]]]
            ) -> Value:
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
      elements = tuple(elements[0])
    if not elements:
      raise ValueError("array requires at least one element")
    coerced: List[Value] = []
    element_type: Optional[str] = None
    for element in elements:
      value = self._coerce_array_element(element, element_type)
      if element_type is None:
        element_type = value.typ
      coerced.append(value)
    assert element_type is not None
    name = self._next_value()
    operand_names = ", ".join(value.name for value in coerced)
    array_type = f"!lpn.array<{element_type}>"
    operand_types = ", ".join(value.typ for value in coerced)
    self._append(f"{name} = lpn.array {operand_names} : {operand_types} -> {array_type}")
    return self._wrap_value(name, array_type)

  def array_alloc(self, size: Union[Value, int], fill_value: Union[Value, int, float]) -> Value:
    sz = self._coerce_value(size, "index")
    fill = self._coerce_array_element(fill_value)
    name = self._next_value()
    typ = f"!lpn.array<{fill.typ}>"
    self._append(f"{name} = lpn.array.alloc {sz.name}, {fill.name} : {sz.typ}, {fill.typ} -> {typ}")
    return self._wrap_value(name, typ)


  def array_get(self,
                array_value: Value,
                index: Union[Value, int]) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_get expects an !lpn.array value")
    idx = self._coerce_value(index, "index")
    element_type = self._array_element_type(array_value.typ)
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.get {array_value.name}, {idx.name} : ({array_value.typ}, index) -> {element_type}")
    return self._wrap_value(name, element_type)

  def key_literal(self, name: str) -> KeyValue:
    existing = self._literal_keys.get(name)
    if existing is not None:
      return existing
    value = self._next_value()
    self._append(
        f"{value} = lpn.key.literal \"{name}\" : !lpn.key")
    key = self._wrap_key(value)
    self._literal_keys[name] = key
    return key

  def key_reg(self, identifier: Union[Value, int]) -> KeyValue:
    key_id = identifier
    if not isinstance(key_id, Value) or key_id.typ != "i64":
      key_id = self._coerce_value(key_id, "i64")
    name = self._next_value()
    self._append(
        f"{name} = lpn.key.reg {key_id.name} : i64 -> !lpn.key")
    return self._wrap_key(name)

  def _get_literal_segment_value(self, literal: str) -> Optional[Value]:
    if not literal:
      return None
    existing = self._literal_segment_values.get(literal)
    if existing is not None:
      return existing
    hashed = self._literal_segment_hashes.get(literal)
    if hashed is None:
      hashed = _hash_literal_segment_to_i64(literal)
      self._literal_segment_hashes[literal] = hashed
    value = self.const_i64(hashed)
    self._literal_segment_values[literal] = value
    return value

  def _materialize_dynamic_key(self,
                               segments: Sequence[Union[str, Value, int]]
                               ) -> KeyValue:
    segment_values: List[Value] = []
    for segment in segments:
      if isinstance(segment, str):
        literal_value = self._get_literal_segment_value(segment)
        if literal_value is None:
          continue
        segment_values.append(literal_value)
      else:
        segment_values.append(self._coerce_value(segment, "i64"))
    if not segment_values:
      raise ValueError("dynamic key requires at least one runtime segment")
    array_value = self.array(segment_values)
    helper_name = self._ensure_dynamic_key_helper()
    if helper_name is None:
      current = self.const_i64(_FNV_OFFSET_BASIS_I64)
      prime_value = self.const_i64(_FNV_PRIME)
      for value in segment_values:
        mixed = self.xori(current, value)
        current = self.muli(mixed, prime_value)
      return self.key_reg(current)
    name = self._next_value()
    self._append(
        f"{name} = func.call @{helper_name}({array_value.name}) : ({array_value.typ}) -> !lpn.key")
    return self._wrap_key(name)

  def _ensure_dynamic_key_helper(self) -> Optional[str]:
    if self._owner is None:
      return None
    return self._owner.ensure_dynamic_key_helper()

  def _ensure_key(self,
                  key: Union[str, KeyValue, Value, int]) -> KeyValue:
    if isinstance(key, KeyValue):
      return key
    if isinstance(key, Value):
      if key.typ == "!lpn.key":
        return KeyValue(self, key.name)
      return self.key_reg(key)
    if isinstance(key, int):
      return self.key_reg(key)
    if isinstance(key, str):
      dynamic_segments = self._match_dynamic_key_literal(key)
      if dynamic_segments is not None:
        return self._materialize_dynamic_key(dynamic_segments)
      return self.key_literal(key)
    raise TypeError("expected a key literal string, KeyValue, Value, or int")

  def _match_dynamic_key_literal(
      self, literal: str) -> Optional[List[Union[str, Value]]]:
    """Detect literal strings that embed SSA value names via f-strings."""
    matches = list(TransitionBuilder._value_name_re.finditer(literal))
    if not matches:
      return None
    segments: List[Union[str, Value]] = []
    cursor = 0
    for match in matches:
      start, end = match.span()
      if start > cursor:
        prefix = literal[cursor:start]
        if prefix:
          segments.append(prefix)
      value_name = match.group(0)
      value = self._ssa_values.get(value_name)
      if value is None:
        return None
      segments.append(value)
      cursor = end
    if cursor < len(literal):
      suffix = literal[cursor:]
      if suffix:
        segments.append(suffix)
    if not any(isinstance(segment, Value) for segment in segments):
      return None
    return segments

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

  def muli(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.muli {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def xori(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.xori {lhs_val}, {rhs_val} : {typ}")
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

  def select(self,
             cond: Value,
             true_value: Union[Value, int, float],
             false_value: Union[Value, int, float],
             *,
             typ: Optional[str] = None) -> Value:
    if not isinstance(cond, Value) or cond.typ != "i1":
      raise TypeError("select condition must be an i1 Value")
    value_type = typ
    if value_type is None:
      if isinstance(true_value, Value):
        value_type = true_value.typ
      elif isinstance(false_value, Value):
        value_type = false_value.typ
      else:
        value_type = "i64"
    true_val = self._coerce_value(true_value, value_type)
    false_val = self._coerce_value(false_value, value_type)
    name = self._next_value()
    self._append(
        f"{name} = arith.select {cond.name}, {true_val.name}, {false_val.name} : {value_type}")
    return self._wrap_value(name, value_type)

  def _capture_ops(
      self,
      fn: Optional[Callable[..., None]],
      *fn_args: Any
  ) -> List[Statement]:
    if fn is None:
      return []
    saved_ops = self._ops
    branch_ops: List[Statement] = []
    self._ops = branch_ops
    try:
      fn(self, *fn_args)
    finally:
      self._ops = saved_ops
    return branch_ops

  def _capture_ops_and_results(
      self,
      fn: Optional[Callable[..., Any]],
      *fn_args: Any
  ) -> Tuple[List[Statement], Any]:
    if fn is None:
      return [], None
    saved_ops = self._ops
    branch_ops: List[Statement] = []
    self._ops = branch_ops
    result = None
    try:
      result = fn(self, *fn_args)
    finally:
      self._ops = saved_ops
    return branch_ops, result

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

  def divi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64",
           signed: bool = True) -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    op = "arith.divsi" if signed else "arith.divui"
    self._append(f"{name} = {op} {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def remi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64",
           signed: bool = True) -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    op = "arith.remsi" if signed else "arith.remui"
    self._append(f"{name} = {op} {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

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

  def token_get(self,
                token: TokenValue,
                key: Union[str, RuntimeKey, int]
                ) -> Value:
    key_value = self._ensure_key(key)
    name = self._next_value()
    self._append(
        f"{name} = lpn.token.get {token.name}, {key_value.name} : !lpn.token, !lpn.key -> i64")
    return self._wrap_value(name, "i64")

  def token_set(self,
                token: TokenValue,
                key: Union[str, KeyValue, Value, int],
                value_ssa: Union[Value, int]) -> TokenValue:
    key_value = self._ensure_key(key)
    value = self._coerce_value(value_ssa, "i64")
    name = self._next_value()
    self._append(
        f"{name} = lpn.token.set {token.name}, {key_value.name}, {value.name} : !lpn.token, !lpn.key, i64 -> !lpn.token")
    return self._wrap_token(name)

  def clone(self, token: TokenValue) -> TokenValue:
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.clone\"({token.name}) : (!lpn.token) -> !lpn.token")
    return self._wrap_token(name)

  def create(self, properties: Optional[Dict[str, int]] = None) -> TokenValue:
    props_dict = properties or {}
    props = ", ".join(
        f"{key} = {value} : i64" for key, value in sorted(props_dict.items()))
    attr = f"{{{props}}}" if props else "{}"
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.create\"() {{log_prefix = {attr}}} : () -> !lpn.token")
    return self._wrap_token(name)

  def reg(self, key: Union[str, KeyValue, Value, int]) -> KeyValue:
    """Creates a key handle from literal strings, register ids, or dynamic expressions."""
    return self._ensure_key(key)

  def array_set(self,
                array_value: Value,
                index: Union[Value, int],
                value: Union[Value, int, float]) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_set expects an !lpn.array value")
    idx = self._coerce_value(index, "index")
    element_type = self._array_element_type(array_value.typ)
    val = self._coerce_array_element(value, element_type)
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.set {array_value.name}, {idx.name}, {val.name} : ({array_value.typ}, index, {element_type}) -> {array_value.typ}")
    return self._wrap_value(name, array_value.typ)

  def array_len(self, array_value: Value) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_len expects an !lpn.array value")
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.len {array_value.name} : {array_value.typ} -> index")
    return self._wrap_value(name, "index")

  def materialize_place(self, place: PlaceHandle) -> None:
    self._ensure_place_handle(place)

  def for_range(self,
                lower: Union[Value, int],
                upper: Union[Value, int],
                *,
                step: Union[Value, int] = 1,
                iter_args: Sequence[Value] = (),
                body: Callable[..., Any]) -> Sequence[Value]:
    lb = self._coerce_value(lower, "index")
    ub = self._coerce_value(upper, "index")
    st = self._coerce_value(step, "index")
    iv_name = self._next_value()
    iv_value = self._wrap_value(iv_name, "index")

    # Prepare iter_args for ForStatement
    iter_args_info = []
    for arg in iter_args:
      iter_args_info.append((self._next_value(), arg.name, arg.typ))

    # Prepare block args for body
    block_args = [iv_value]
    for name, _, typ in iter_args_info:
      block_args.append(self._wrap_value(name, typ))

    body_ops, body_results = self._capture_ops_and_results(body, *block_args)

    yield_values = []
    yield_types = []
    if iter_args:
      if not isinstance(body_results, (list, tuple)):
        body_results = [body_results]
      if len(body_results) != len(iter_args):
        raise ValueError(f"for_range body must return {len(iter_args)} values, got {len(body_results)}")
      for res in body_results:
        if not isinstance(res, Value):
           raise TypeError(f"for_range body must return Values, got {type(res)}")
        yield_values.append(res.name)
        yield_types.append(res.typ)

    results = []
    result_names = []
    if iter_args:
      for _, _, typ in iter_args_info:
        name = self._next_value()
        result_names.append(name)
        results.append(self._wrap_value(name, typ))

    self._ops.append(
        ForStatement(iv_name, lb.name, ub.name, st.name, body_ops,
                     iter_args=iter_args_info,
                     results=result_names,
                     yield_values=yield_values,
                     yield_types=yield_types))

    if len(results) == 1:
      return results[0]
    return tuple(results)

  def _call_helper(self,
                   function: "FunctionDef",
                   args: Sequence[Any]) -> Optional[Union[Value, TokenValue, Tuple[Union[Value, TokenValue], ...]]]:
    arg_types = [arg.typ for arg in function.builder.arguments]
    if len(args) != len(arg_types):
      raise TypeError(
          f"helper '{function.name}' expected {len(arg_types)} arguments, got {len(args)}")
    operands = [
        self._coerce_function_argument(value, typ)
        for value, typ in zip(args, arg_types)
    ]
    operand_names = ", ".join(value.name for value in operands)
    operand_types = ", ".join(value.typ for value in operands)
    result_types = list(function.builder.result_types)
    result_sig = ", ".join(result_types)
    result_names: List[str] = []
    assign_prefix = ""
    if result_types:
      result_names = [self._next_value() for _ in result_types]
      assigned = ", ".join(result_names)
      assign_prefix = f"{assigned} = "
    self._append(
        f"{assign_prefix}func.call @{function.name}({operand_names}) : ({operand_types}) -> ({result_sig})")
    results: List[Union[Value, TokenValue]] = []
    for name, typ in zip(result_names, result_types):
      if typ == "!lpn.token":
        results.append(self._wrap_token(name))
      else:
        results.append(self._wrap_value(name, typ))
    if not results:
      return None
    if len(results) == 1:
      return results[0]
    return tuple(results)

  def render(self) -> List[str]:
    ops = []
    if self._ops:
      for op in self._ops:
        ops.append(op.render("        "))
    ops.append("        lpn.schedule.return")
    return ops


@dataclass
class TransitionDef:
  name: str
  builder: TransitionBuilder
  fn: Callable[..., Any]
  script: bool
  jit: bool


@dataclass
class FunctionDef:
  name: str
  builder: "FuncBuilder"
  visibility: str


class FuncBuilder(TransitionBuilder):
  def __init__(self,
               name: str,
               owner: Optional["NetBuilder"],
               arg_types: Sequence[str],
               result_types: Sequence[str],
               arg_names: Optional[Sequence[str]] = None):
    super().__init__(name, owner=owner)
    self._result_types = list(result_types)
    self._arguments: List[Value] = []
    for idx, typ in enumerate(arg_types):
      if arg_names and idx < len(arg_names):
        arg_name = arg_names[idx]
      else:
        arg_name = f"%arg{idx}"
      self._arguments.append(self._register_argument(arg_name, typ))
    self._func_return_emitted = False

  @property
  def arguments(self) -> List[Value]:
    return self._arguments

  @property
  def result_types(self) -> List[str]:
    return self._result_types

  def func_return(self, *values: Value) -> None:
    if not values:
      if self._result_types:
        raise ValueError("func.return requires values for non-void functions")
      self._append("func.return")
      self._func_return_emitted = True
      return
    if len(values) != len(self._result_types):
      raise ValueError(
          f"func.return expected {len(self._result_types)} values, got {len(values)}")
    operand_names = ", ".join(value.name for value in values)
    operand_types = ", ".join(value.typ for value in values)
    self._append(f"func.return {operand_names} : {operand_types}")
    self._func_return_emitted = True

  def render(self) -> List[str]:
    ops = []
    if self._ops:
      for op in self._ops:
        ops.append(op.render("        "))
    if not self._func_return_emitted:
      if self._result_types:
        raise ValueError(
            f"function '{self.name}' is missing a func.return statement")
      ops.append("        func.return")
    return ops


class NetBuilder:
  """Tiny DSL for emitting the new MLIR dialect."""

  _TYPE_ALIASES = {
      "token": "!lpn.token",
      "tokenvalue": "!lpn.token",
      "token_value": "!lpn.token",
      "place": "!lpn.place",
      "placehandle": "!lpn.place",
      "key": "!lpn.key",
      "runtimekey": "!lpn.key",
      "i64": "i64",
      "index": "index",
      "f64": "f64",
  }

  def __init__(self, name: str = "net"):
    self.name = name
    self._places: List[tuple[PlaceHandle, Optional[int], Optional[int], bool]] = []
    self._transitions: List[TransitionDef] = []
    self._helper_functions: Dict[str, List[str]] = {}
    self._functions: List["FunctionDef"] = []
    self._function_map: Dict[str, "FunctionDef"] = {}

  def _make_helper_stub(self,
                        builder: TransitionBuilder,
                        function: "FunctionDef") -> Callable[..., Any]:
    def _helper(*args: Any) -> Any:
      return builder._call_helper(function, args)
    _helper.__name__ = function.name
    _helper.__qualname__ = function.name
    return _helper

  def _register_function_def(self, function: "FunctionDef") -> None:
    if function.name in self._function_map:
      raise ValueError(f"helper function '{function.name}' already defined")
    self._functions.append(function)
    self._function_map[function.name] = function

  def _unregister_function_def(self, function: "FunctionDef") -> None:
    existing = self._function_map.get(function.name)
    if existing is function:
      self._function_map.pop(function.name, None)
    if function in self._functions:
      self._functions.remove(function)

  def _bind_helper_stubs(self, builder: TransitionBuilder) -> None:
    if not self._functions:
      return
    for function in self._functions:
      setattr(builder, function.name, self._make_helper_stub(builder, function))

  def _normalize_type_spec(self, spec: Any) -> str:
    if isinstance(spec, str):
      cleaned = spec.strip()
      alias_key = cleaned.lower().replace(" ", "")
      array_match = re.fullmatch(r"list\[(.+)\]", alias_key)
      if array_match:
        inner = self._normalize_type_spec(array_match.group(1))
        return f"!lpn.array<{inner}>"
      if cleaned.startswith("!lpn") or cleaned in ("i64", "index", "f64"):
        return cleaned
      alias_value = NetBuilder._TYPE_ALIASES.get(alias_key)
      if alias_value:
        return alias_value
      raise ValueError(
          f"unsupported helper type alias '{spec}'; expected full MLIR type or one of {sorted(NetBuilder._TYPE_ALIASES.keys())}")
    origin = get_origin(spec)
    if origin in (list, List, Sequence, tuple):
      args = get_args(spec)
      if len(args) != 1:
        raise TypeError("List[...] helper aliases must have one argument")
      inner = self._normalize_type_spec(args[0])
      return f"!lpn.array<{inner}>"
    if spec in (TokenValue, Value):
      return "!lpn.token"
    if spec is PlaceHandle:
      return "!lpn.place"
    raise TypeError(
        f"unsupported helper type specification {spec!r}")

  def place(self,
            name: str,
            *,
            capacity: Optional[int] = None,
            initial_tokens: Optional[int] = None,
            observable: bool = False) -> PlaceHandle:
    handle = PlaceHandle(name)
    self._places.append((handle, capacity, initial_tokens, observable))
    return handle

  def _run_jit_script(self,
                      fn: Callable[[TransitionBuilder], None],
                      builder: TransitionBuilder,
                      *,
                      require_builder_arg: bool,
                      extra_args: Sequence[Any] = ()) -> None:
    executor = TransitionScriptExecutor(fn,
                                        require_builder_arg=require_builder_arg)
    executor(builder, *extra_args)

  def _make_transition(self,
                       name: str,
                       *,
                       script: bool,
                       jit: bool
                       ) -> Callable[[Callable[[TransitionBuilder], None]], Callable[[TransitionBuilder], None]]:
    def decorator(fn: Callable[[TransitionBuilder], None]):
      builder = TransitionBuilder(name, owner=self)
      self._bind_helper_stubs(builder)
      if script:
        self._run_jit_script(fn,
                             builder,
                             require_builder_arg=not jit)
      else:
        fn(builder)
      definition = TransitionDef(name=name,
                                 builder=builder,
                                 fn=fn,
                                 script=script,
                                 jit=jit)
      self._transitions.append(definition)
      return fn
    return decorator

  def transition(self, name: Optional[Union[str, Callable]] = None):
    """Decorator for defining scheduler transitions."""
    if callable(name):
      fn = name  # type: ignore[assignment]
      actual = fn.__name__
      return self._make_transition(actual, script=True, jit=True)(fn)

    def _decorator(fn: Callable[[TransitionBuilder], None]):
      actual = name if isinstance(name, str) else fn.__name__
      return self._make_transition(actual, script=True, jit=True)(fn)

    return _decorator

  def jit(self, name: Optional[Union[str, Callable]] = None):
    """Backward-compatible alias for transition()."""
    return self.transition(name)

  def func(self,
           name: Optional[Union[str, Callable]] = None,
           *,
           args: Sequence[Union[str, Tuple[str, Any]]] = (),
           results: Sequence[Any] = (),
           private: bool = True):
    """Define a helper MLIR `func.func` using the DSL."""

    def normalize_args(
        specs: Sequence[Union[str, Tuple[str, Any]]]
    ) -> Tuple[List[str], List[str]]:
      labels: List[str] = []
      types: List[str] = []
      used: Dict[str, int] = {}
      for idx, spec in enumerate(specs):
        if isinstance(spec, tuple):
          label, typ = spec
        else:
          label, typ = (f"arg{idx}", spec)
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", label)
        if not sanitized:
          sanitized = f"arg{idx}"
        if sanitized[0].isdigit():
          sanitized = f"_{sanitized}"
        base = sanitized
        suffix = used.get(base, 0)
        candidate = sanitized
        while candidate in used:
          suffix += 1
          candidate = f"{base}_{suffix}"
        used[candidate] = 1
        labels.append(candidate)
        types.append(self._normalize_type_spec(typ))
      return labels, types

    def build(actual_name: str,
              fn: Callable[[TransitionBuilder], None]) -> Callable[[TransitionBuilder], None]:
      arg_labels, arg_types = normalize_args(args)
      ssa_names = [f"%{label}" for label in arg_labels]
      builder = FuncBuilder(actual_name,
                            owner=self,
                            arg_types=arg_types,
                            result_types=[self._normalize_type_spec(r)
                                          for r in results],
                            arg_names=ssa_names)
      visibility = "private" if private else "public"
      function_def = FunctionDef(actual_name, builder, visibility)
      self._register_function_def(function_def)
      try:
        self._bind_helper_stubs(builder)
        self._run_jit_script(fn,
                             builder,
                             require_builder_arg=False,
                             extra_args=builder.arguments)
      except Exception:
        self._unregister_function_def(function_def)
        raise

      def wrapper(*args, **kwargs):
        builder = None
        for arg in args:
          if isinstance(arg, (Value, TokenValue, KeyValue)):
            builder = arg.builder
            break
        if builder is None:
          for arg in kwargs.values():
            if isinstance(arg, (Value, TokenValue, KeyValue)):
              builder = arg.builder
              break

        if builder is not None:
          if kwargs:
            raise TypeError("keyword arguments are not supported in JIT helper calls")
          return builder._call_helper(function_def, args)

        try:
          return fn(*args, **kwargs)
        except FunctionReturn as e:
          return e.value

      return wrapper

    if callable(name):
      fn = name  # type: ignore[assignment]
      actual = fn.__name__
      return build(actual, fn)

    def _decorator(fn: Callable[[TransitionBuilder], None]):
      actual = name if isinstance(name, str) else fn.__name__
      return build(actual, fn)

    return _decorator

  def python_simulator(self) -> "PythonSimulator":
    """Create a lightweight interpreter that can run transitions in Python."""
    return PythonSimulator(self)

  def ensure_dynamic_key_helper(self) -> str:
    symbol = "__lpn_hash_key"
    if symbol in self._helper_functions:
      return symbol
    for func in self._functions:
      if func.name == symbol:
        return symbol
    if symbol not in self._helper_functions:
      self._helper_functions[symbol] = self._render_dynamic_key_helper(symbol)
    return symbol

  def _render_dynamic_key_helper(self, name: str) -> List[str]:
    offset = _FNV_OFFSET_BASIS_I64
    prime = _FNV_PRIME
    return [
        f"func.func private @{name}(%segments: !lpn.array<i64>) -> !lpn.key {{",
        "  %c0 = arith.constant 0 : index",
        "  %c1 = arith.constant 1 : index",
        "  %len = lpn.array.len %segments : !lpn.array<i64> -> index",
        f"  %offset = arith.constant {offset} : i64",
        f"  %prime = arith.constant {prime} : i64",
        "  %result = scf.for %iv = %c0 to %len step %c1 iter_args(%acc = %offset) -> (i64) {",
        "    %chunk = lpn.array.get %segments, %iv : (!lpn.array<i64>, index) -> i64",
        "    %xor = arith.xori %acc, %chunk : i64",
        "    %mul = arith.muli %xor, %prime : i64",
        "    scf.yield %mul : i64",
        "  }",
        "  %key = lpn.key.reg %result : i64 -> !lpn.key",
        "  return %key : !lpn.key",
        "}",
    ]

  def build(self) -> str:
    lines = ["module {"]

    lines.append("  lpn.net {")

    if self._helper_functions:
      for name in sorted(self._helper_functions.keys()):
        helper_lines = self._helper_functions[name]
        for line in helper_lines:
          lines.append(f"    {line}")
      lines.append("")

    if self._functions:
      for func in self._functions:
        visibility_kw = f"{func.visibility} " if func.visibility else ""
        arg_sig = ", ".join(
            f"{arg.name}: {arg.typ}" for arg in func.builder.arguments)
        result_types = func.builder.result_types
        result_clause = ""
        if result_types:
          joined = ", ".join(result_types)
          result_clause = f" -> ({joined})"
        lines.append(
            f"    func.func {visibility_kw}@{func.name}({arg_sig}){result_clause} {{")
        block_sig = arg_sig
        block_header = f"      ^bb0({block_sig}):" if block_sig else "      ^bb0:"
        lines.append(block_header)
        for op in func.builder.render():
          lines.append(op)
        lines.append("    }")
        lines.append("")

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

    for definition in self._transitions:
      transition = definition.builder
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


@dataclass(frozen=True)
class RuntimeKey:
  identifier: Any

  def __str__(self) -> str:
    return str(self.identifier)


def _normalize_runtime_key(key: Union[str, RuntimeKey, int]) -> Any:
  if isinstance(key, RuntimeKey):
    return key.identifier
  if isinstance(key, (str, int)):
    return key
  raise TypeError(f"unsupported runtime key type {type(key)}")


class RuntimeToken:
  def __init__(self, fields: Optional[Dict[Any, int]] = None):
    self._fields: Dict[Any, int] = dict(fields or {})

  def __repr__(self) -> str:
    return f"RuntimeToken({self._fields})"

  def get(self, key: Union[str, RuntimeKey, int]) -> int:
    return self._fields.get(_normalize_runtime_key(key), 0)

  def set(self, key: Union[str, RuntimeKey, int], value: Union[int, float]) -> "RuntimeToken":
    updated = dict(self._fields)
    updated[_normalize_runtime_key(key)] = int(value)
    return RuntimeToken(updated)

  def clone(self) -> "RuntimeToken":
    return RuntimeToken(self._fields)

  def to_dict(self) -> Dict[Any, int]:
    return dict(self._fields)


class _TransitionBlocked(RuntimeError):
  def __init__(self, place: str):
    super().__init__(f"place '{place}' is empty")
    self.place = place


class _RuntimePlace:
  def __init__(self,
               name: str,
               *,
               capacity: Optional[int] = None,
               initial_tokens: int = 0):
    self.name = name
    self.capacity = capacity
    self.tokens: deque[RuntimeToken] = deque(
        RuntimeToken() for _ in range(initial_tokens))
    self._scheduled: List[tuple[float, int, RuntimeToken]] = []
    self._sequence = itertools.count()

  def __len__(self) -> int:
    return len(self.tokens)

  def pushleft(self, token: RuntimeToken) -> None:
    self.tokens.appendleft(token)

  def push(self, token: RuntimeToken) -> None:
    self.tokens.append(token)

  def pop(self) -> RuntimeToken:
    if not self.tokens:
      raise _TransitionBlocked(self.name)
    return self.tokens.popleft()

  def schedule(self, token: RuntimeToken, ready_time: float) -> None:
    event = (ready_time, next(self._sequence), token)
    heapq.heappush(self._scheduled, event)

  def commit_ready(self, upto_time: float) -> bool:
    changed = False
    while self._scheduled and self._scheduled[0][0] <= upto_time:
      _, _, token = heapq.heappop(self._scheduled)
      self.tokens.append(token)
      changed = True
    return changed

  def earliest_ready(self) -> Optional[float]:
    if not self._scheduled:
      return None
    return self._scheduled[0][0]

  def scheduled_len(self) -> int:
    return len(self._scheduled)


class _TransitionContext:
  def __init__(self):
    self.taken: List[tuple[_RuntimePlace, RuntimeToken]] = []
    self.emits: List[tuple[_RuntimePlace, RuntimeToken, float]] = []


class PythonRuntimeAPI:
  """Implements the DSL surface for the Python simulator."""

  def __init__(self, simulator: "PythonSimulator"):
    self._sim = simulator

  def _as_int(self, value: Union[int, float]) -> int:
    return int(value)

  def take(self, place: PlaceHandle) -> RuntimeToken:
    return self._sim._transaction_take(place)

  def take_handle(self, handle: PlaceHandle) -> RuntimeToken:
    return self.take(handle)

  def emit(self,
           place: PlaceHandle,
           token: RuntimeToken,
           delay: Optional[Union[int, float]] = None) -> None:
    self._sim._transaction_emit(place, token, delay)

  def emit_handle(self,
                  handle: PlaceHandle,
                  token: RuntimeToken,
                  delay: Optional[Union[int, float]] = None) -> None:
    self.emit(handle, token, delay)

  def count(self, place: PlaceHandle) -> int:
    return self._sim._count(place)

  def array(self,
            *elements: Union[Value, PlaceHandle, int, float,
                             Sequence[Union[Value, PlaceHandle, int, float]]]
            ) -> tuple:
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
      elements = tuple(elements[0])
    return tuple(elements)

  def array_alloc(self, size: Union[int, Value], fill_value: Any) -> list:
    sz = int(size)
    return [fill_value for _ in range(sz)]

  def array_set(self, array_value: Sequence[Any], index: int, value: Any) -> list:
    idx = int(index)
    lst = list(array_value)
    lst[idx] = value
    return lst

  def array_get(self,
                array_value: Sequence[Any],
                index: Union[int, Value]) -> Any:
    if isinstance(index, Value):
      raise TypeError("runtime arrays expect concrete indices")
    return array_value[index]

  def array_len(self, array_value: Sequence[Any]) -> int:
    return len(array_value)

  def create(self,
                   properties: Optional[Dict[str, int]] = None) -> RuntimeToken:
    props = dict(properties or {})
    return RuntimeToken(props)

  def clone(self, token: RuntimeToken) -> RuntimeToken:
    return token.clone()

  def token_get(self,
                token: RuntimeToken,
                key: Union[str, RuntimeKey, int]) -> int:
    return token.get(key)

  def token_set(self,
                token: RuntimeToken,
                key: Union[str, RuntimeKey, int],
                value: Union[int, float]) -> RuntimeToken:
    return token.set(key, value)

  def key_literal(self, name: str) -> RuntimeKey:
    return RuntimeKey(name)

  def key_reg(self, identifier: Union[int, float]) -> RuntimeKey:
    return RuntimeKey(("reg", int(identifier)))

  def reg(self, key: Union[str, RuntimeKey, int]) -> RuntimeKey:
    if isinstance(key, RuntimeKey):
      return key
    if isinstance(key, str):
      return RuntimeKey(key)
    return RuntimeKey(("reg", int(key)))

  def const_i64(self, value: int) -> int:
    return int(value)

  def i64(self, value: int) -> int:
    return self.const_i64(value)

  def const_index(self, value: int) -> int:
    return int(value)

  def index(self, value: int) -> int:
    return self.const_index(value)

  def const_f64(self, value: float) -> float:
    return float(value)

  def f64(self, value: float) -> float:
    return self.const_f64(value)

  def addi(self, lhs: Union[int, float], rhs: Union[int, float], *,
           typ: str = "i64") -> Union[int, float]:
    return lhs + rhs

  def subi(self, lhs: Union[int, float], rhs: Union[int, float], *,
           typ: str = "i64") -> Union[int, float]:
    return lhs - rhs

  def muli(self, lhs: Union[int, float], rhs: Union[int, float], *,
           typ: str = "i64") -> Union[int, float]:
    return lhs * rhs

  def xori(self, lhs: int, rhs: int, *, typ: str = "i64") -> int:
    return int(lhs) ^ int(rhs)

  def addf(self, lhs: float, rhs: float, *, typ: str = "f64") -> float:
    return float(lhs) + float(rhs)

  def subf(self, lhs: float, rhs: float, *, typ: str = "f64") -> float:
    return float(lhs) - float(rhs)

  def divf(self, lhs: float, rhs: float, *, typ: str = "f64") -> float:
    return float(lhs) / float(rhs)

  def divi(self,
           lhs: Union[int, float],
           rhs: Union[int, float],
           *,
           typ: str = "i64",
           signed: bool = True) -> int:
    return int(lhs) // int(rhs)

  def sitofp(self,
             value: Union[int, float],
             *,
             src_type: str = "i64",
             dst_type: str = "f64") -> float:
    return float(value)

  def index_cast(self,
                 value: Union[int, float],
                 *,
                 src_type: str = "index",
                 dst_type: str = "i64") -> int:
    return int(value)

  def for_range(self,
                lower: Union[int, float],
                upper: Union[int, float],
                *,
                step: Union[int, float] = 1,
                body: Callable [['PythonRuntimeAPI', int], None]) -> None:
    lb = int(lower)
    ub = int(upper)
    st = int(step)
    for idx in range(lb, ub, st):
      body(self, idx)

  def if_op(self,
            cond: Union[bool, int],
            true_fn: Callable [['PythonRuntimeAPI'], None],
            false_fn: Optional[Callable [['PythonRuntimeAPI'], None]] = None) -> None:
    if bool(cond):
      true_fn(self)
    elif false_fn:
      false_fn(self)

  def func_return(self, *values: Any) -> None:
    if not values:
      raise FunctionReturn(None)
    if len(values) == 1:
      raise FunctionReturn(values[0])
    raise FunctionReturn(values)

  def exported_symbols(self) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for attr in dir(self):
      if attr.startswith("_"):
        continue
      value = getattr(self, attr)
      if callable(value):
        mapping[attr] = value
    mapping["builder"] = self
    return mapping


class PythonSimulator:
  """Event-driven interpreter that replays transitions directly in Python."""

  def __init__(self, net: NetBuilder):
    self._place_specs = list(net._places)
    self._transitions = list(net._transitions)
    self._transition_map = {definition.name: definition
                            for definition in self._transitions}
    self._current_time: float = 0.0
    self._active_transaction: Optional[_TransitionContext] = None
    self.reset()

  @property
  def current_time(self) -> float:
    return self._current_time

  def reset(self) -> None:
    self._places: Dict[str, _RuntimePlace] = {}
    for handle, capacity, initial, _observable in self._place_specs:
      initial_count = int(initial or 0)
      self._places[handle.name] = _RuntimePlace(handle.name,
                                                capacity=capacity,
                                                initial_tokens=initial_count)
    self._current_time = 0.0
    self._active_transaction = None
    self._commit_ready_tokens(self._current_time)

  def _runtime_place(self, place: Union[PlaceHandle, str]) -> _RuntimePlace:
    key = place.name if isinstance(place, PlaceHandle) else place
    runtime_place = self._places.get(key)
    if runtime_place is None:
      raise KeyError(f"unknown place '{key}'")
    return runtime_place

  def _begin_transaction(self) -> _TransitionContext:
    if self._active_transaction is not None:
      raise RuntimeError("nested transitions are not supported")
    self._active_transaction = _TransitionContext()
    return self._active_transaction

  def _require_transaction(self) -> _TransitionContext:
    if self._active_transaction is None:
      raise RuntimeError("transition context is not active")
    return self._active_transaction

  def _transaction_take(self, place: PlaceHandle) -> RuntimeToken:
    ctx = self._require_transaction()
    runtime_place = self._runtime_place(place)
    token = runtime_place.pop()
    ctx.taken.append((runtime_place, token))
    return token

  def _transaction_emit(self,
                        place: PlaceHandle,
                        token: RuntimeToken,
                        delay: Optional[Union[int, float]]) -> None:
    ctx = self._require_transaction()
    runtime_place = self._runtime_place(place)
    delay_value = 0.0 if delay is None else float(delay)
    if delay_value < 0:
      delay_value = 0.0
    ready_time = self._current_time + delay_value
    ctx.emits.append((runtime_place, token, ready_time))

  def _commit_transaction(self) -> None:
    ctx = self._require_transaction()
    for runtime_place, token, ready_time in ctx.emits:
      runtime_place.schedule(token, ready_time)
    self._active_transaction = None
    self._commit_ready_tokens(self._current_time)

  def _rollback_transaction(self) -> None:
    ctx = self._require_transaction()
    for runtime_place, token in reversed(ctx.taken):
      runtime_place.pushleft(token)
    self._active_transaction = None

  def _take(self, place: PlaceHandle) -> RuntimeToken:
    runtime_place = self._runtime_place(place)
    token = runtime_place.pop()
    return token

  def _emit(self,
            place: PlaceHandle,
            token: RuntimeToken,
            delay: Optional[Union[int, float]] = None) -> None:
    runtime_place = self._runtime_place(place)
    delay_value = 0.0 if delay is None else float(delay)
    if delay_value < 0:
      delay_value = 0.0
    ready_time = self._current_time + delay_value
    runtime_place.schedule(token, ready_time)
    self._commit_ready_tokens(self._current_time)

  def _count(self, place: PlaceHandle) -> int:
    self._commit_ready_tokens(self._current_time)
    runtime_place = self._runtime_place(place)
    return len(runtime_place)

  def _commit_ready_tokens(self, upto_time: float) -> bool:
    changed = False
    for runtime_place in self._places.values():
      changed |= runtime_place.commit_ready(upto_time)
    return changed

  def _earliest_scheduled_time(self) -> Optional[float]:
    earliest: Optional[float] = None
    for runtime_place in self._places.values():
      candidate = runtime_place.earliest_ready()
      if candidate is None:
        continue
      if earliest is None or candidate < earliest:
        earliest = candidate
    return earliest

  def _has_pending_scheduled_tokens(self) -> bool:
    return any(place.scheduled_len() > 0 for place in self._places.values())

  def place_contents(self,
                     place: Union[str, PlaceHandle]) -> List[RuntimeToken]:
    self._commit_ready_tokens(self._current_time)
    runtime_place = self._runtime_place(place)
    return list(runtime_place.tokens)

  def place_dicts(self,
                  place: Union[str, PlaceHandle]) -> List[Dict[Any, int]]:
    return [token.to_dict() for token in self.place_contents(place)]

  def fire(self,
           transition_name: str,
           *args: Any,
           repeats: int = 1,
           **kwargs: Any) -> None:
    definition = self._transition_map.get(transition_name)
    if not definition:
      raise KeyError(f"transition '{transition_name}' not found")
    for _ in range(repeats):
      fired = self._invoke(definition,
                           *args,
                           allow_block=False,
                           **kwargs)
      if not fired:
        raise RuntimeError(f"transition '{transition_name}' blocked")

  def run(self,
          *,
          max_time: float = 1_000_000.0,
          debug: bool = False,
          reset: bool = True) -> float:
    if reset:
      self.reset()
    else:
      self._commit_ready_tokens(self._current_time)
    while self._current_time <= max_time:
      progress = self._run_all_transitions(debug=debug)
      if progress:
        continue
      next_time = self._earliest_scheduled_time()
      if next_time is None or next_time > max_time:
        break
      self._current_time = next_time
      self._commit_ready_tokens(self._current_time)
    return self._current_time

  def _run_all_transitions(self, *, debug: bool = False) -> bool:
    any_progress = False
    for definition in self._transitions:
      while self._invoke(definition, allow_block=True):
        any_progress = True
        if debug:
          print(f"[t={self._current_time:.3f}] fired {definition.name}")
    return any_progress

  def _invoke(self,
              definition: TransitionDef,
              *args: Any,
              allow_block: bool,
              **kwargs: Any) -> bool:
    api = PythonRuntimeAPI(self)
    self._begin_transaction()
    blocked_reason: Optional[_TransitionBlocked] = None
    injector: Optional[_GlobalNameInjector] = None
    try:
      if definition.jit:
        injector = _GlobalNameInjector(
            definition.fn.__globals__, api.exported_symbols())
        injector.__enter__()
        definition.fn(*args, **kwargs)
      else:
        definition.fn(api, *args, **kwargs)
    except _TransitionBlocked as exc:
      blocked_reason = exc
    except Exception:
      self._rollback_transaction()
      raise
    else:
      self._commit_transaction()
      return True
    finally:
      if injector:
        injector.__exit__(None, None, None)
    self._rollback_transaction()
    if allow_block:
      return False
    place = blocked_reason.place if blocked_reason else "<unknown>"
    raise RuntimeError(
        f"transition '{definition.name}' blocked on place '{place}'")


class FunctionReturn(Exception):
  def __init__(self, value: Any):
    self.value = value


class AssignmentFinder(ast.NodeVisitor):
  def __init__(self):
    self.assigned = set()

  def visit_Assign(self, node: ast.Assign) -> None:
    for target in node.targets:
      if isinstance(target, ast.Name):
        self.assigned.add(target.id)
      elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        self.assigned.add(target.value.id)
    self.generic_visit(node)

  def visit_AugAssign(self, node: ast.AugAssign) -> None:
    if isinstance(node.target, ast.Name):
      self.assigned.add(node.target.id)
    elif isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
      self.assigned.add(node.target.value.id)
    self.generic_visit(node)
