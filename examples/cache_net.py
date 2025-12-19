"""Two-level cache example using a single L1 controller with scf.if."""

from __future__ import annotations

from typing import Iterable

from lpnlang_mlir import NetBuilder, PlaceHandle

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

  def build_line_chain(builder, line_value: str, req, state, idx: int):
    const_idx = builder.const_i64(idx)
    cond = builder.cmpi("eq", line_value, const_idx)

    def hit_branch(b):
      valid = b.token_get(state, f"line{idx}")
      one = b.const_i64(1)
      is_valid = b.cmpi("ne", valid, b.const_i64(0))

      def serve_hit(bb):
        resp = bb.token_set(req, "hit", one)
        delay = bb.const_f64(L1_HIT_DELAY_NS)
        bb.emit(l1_resp, resp, delay=delay)
        bb.emit(l1_state, state)

      def serve_miss(bb):
        miss_token = bb.token_set(req, "hit", bb.const_i64(0))
        bb.emit(l2_req, miss_token)
        bb.emit(l1_state, state)

      b.if_op(is_valid, serve_hit, serve_miss)

    def next_branch(b):
      if idx + 1 < NUM_LINES:
        build_line_chain(b, line_value, req, state, idx + 1)
      else:
        miss_token = b.token_set(req, "hit", b.const_i64(0))
        b.emit(l2_req, miss_token)
        b.emit(l1_state, state)

    builder.if_op(cond, hit_branch, next_branch)

  @net.transition("l1_controller")
  def _(t):
    req = t.take(l1_req)
    state = t.take(l1_state)
    line_val = t.token_get(req, "line")
    t.materialize_place(l1_resp)
    t.materialize_place(l2_req)
    build_line_chain(t, line_val, req, state, 0)

  @net.transition("l2_forward")
  def _(t):
    req = t.take(l2_req)
    delay = t.const_f64(L2_DELAY_NS)
    t.emit(l2_resp, req, delay=delay)

  def build_fill_chain(builder, line_value: str, resp, state, idx: int):
    const_idx = builder.const_i64(idx)
    cond = builder.cmpi("eq", line_value, const_idx)

    def set_line(bb):
      updated = bb.token_set(state, f"line{idx}", bb.const_i64(1))
      delay = bb.const_f64(L1_HIT_DELAY_NS)
      bb.emit(l1_resp, resp, delay=delay)
      bb.emit(l1_state, updated)

    def next_branch(bb):
      if idx + 1 < NUM_LINES:
        build_fill_chain(bb, line_value, resp, state, idx + 1)
      else:
        delay = bb.const_f64(L1_HIT_DELAY_NS)
        bb.emit(l1_resp, resp, delay=delay)
        bb.emit(l1_state, state)

    builder.if_op(cond, set_line, next_branch)

  @net.transition("l1_fill")
  def _(t):
    resp = t.take(l2_resp)
    state = t.take(l1_state)
    line_val = t.token_get(resp, "line")
    t.materialize_place(l1_resp)
    build_fill_chain(t, line_val, resp, state, 0)

  return net.build()


DEFAULT_TRACE = [0x0, 0x10, 0x20, 0x30, 0x0, 0x10]


if __name__ == "__main__":
  print(build_cache_example(DEFAULT_TRACE))
