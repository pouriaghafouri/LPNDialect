"""Two-level cache example using a single L1 controller with scf.if."""

from __future__ import annotations

from typing import Iterable

from lpnlang_mlir import NetBuilder

L1_HIT_DELAY_NS = 10.0
L2_DELAY_NS = 100.0
NUM_LINES = 4
LINE_BYTES = 16


def build_cache_example(addresses: Iterable[int]) -> str:
  net = NetBuilder("cache_net")
  l1_req = net.place("l1_req", observable=True)
  l1_resp = net.place("l1_resp", observable=True)
  l2_req = net.place("l2_req")
  l2_resp = net.place("l2_resp")
  l1_state = net.place("l1_state")
  state_seed = net.place("state_seed", initial_tokens=1)
  issue_ctrl = net.place("issue_ctrl", initial_tokens=1)

  @net.transition("cpu_issue")
  def _(t, trace=list(addresses), req=l1_req, ctrl=issue_ctrl):
    t.take(ctrl)
    for seq, addr in enumerate(trace):
      line = (addr // LINE_BYTES) % NUM_LINES
      token = t.token_create()
      token = t.token_set(token, "addr", t.const_i64(addr))
      token = t.token_set(token, "line", t.const_i64(line))
      token = t.token_set(token, "seq", t.const_i64(seq))
      t.emit(req, token)

  @net.transition("init_state")
  def _(t, seed=state_seed, state_place=l1_state):
    t.take(seed)
    state = t.token_create()
    for idx in range(NUM_LINES):
      state = t.token_set(state, f"line{idx}", t.const_i64(0))
    t.emit(state_place, state)

  def read_line_flag(builder, token, line_idx):
    return builder.token_get(token, ("line", line_idx))

  def write_line_flag(builder, token, line_idx, new_value):
    return builder.token_set(token, ("line", line_idx), new_value)

  @net.transition("l1_controller")
  def _(t):
    req = t.take(l1_req)
    state = t.take(l1_state)
    line_idx = t.token_get(req, "line")
    line_valid = read_line_flag(t, state, line_idx)
    has_line = t.cmpi("ne", line_valid, t.const_i64(0))
    t.materialize_place(l1_resp)
    t.materialize_place(l2_req)

    def serve_hit(b):
      delay = b.const_f64(L1_HIT_DELAY_NS)
      b.emit(l1_resp, req, delay=delay)
      b.emit(l1_state, state)

    def serve_miss(b):
      b.emit(l2_req, req)
      b.emit(l1_state, state)

    t.if_op(has_line, serve_hit, serve_miss)

  @net.transition("l2_forward")
  def _(t):
    req = t.take(l2_req)
    delay = t.const_f64(L2_DELAY_NS)
    t.emit(l2_resp, req, delay=delay)

  @net.transition("l1_fill")
  def _(t):
    resp = t.take(l2_resp)
    state = t.take(l1_state)
    line_idx = t.token_get(resp, "line")
    updated = write_line_flag(t, state, line_idx, t.const_i64(1))
    delay = t.const_f64(L1_HIT_DELAY_NS)
    t.emit(l1_resp, resp, delay=delay)
    t.emit(l1_state, updated)

  return net.build()


DEFAULT_TRACE = [0x0, 0x10, 0x20, 0x30, 0x0, 0x10]


if __name__ == "__main__":
  print(build_cache_example(DEFAULT_TRACE))
