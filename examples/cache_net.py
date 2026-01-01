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

  @net.jit("cpu_issue")
  def cpu_issue(trace=list(addresses)):
    take(issue_ctrl)
    for seq_addr in enumerate(trace):
      seq, addr = seq_addr
      line = (addr // LINE_BYTES) % NUM_LINES
      token = token_create()
      token = token_set(token, "addr", addr)
      token = token_set(token, "line", line)
      token = token_set(token, "seq", seq)
      emit(l1_req, token)

  @net.jit("init_state")
  def init_state():
    take(state_seed)
    state = token_create()
    for idx in range(NUM_LINES):
      state = token_set(state, f"line{idx}", 0)
    emit(l1_state, state)

  @net.jit("l1_controller")
  def l1_controller():
    req = take(l1_req)
    state = take(l1_state)
    line_idx = token_get(req, "line")
    line_valid = token_get(state, ("line", line_idx))
    has_line = line_valid != 0

    if has_line:
      emit(l1_resp, req, delay=L1_HIT_DELAY_NS)
      emit(l1_state, state)
    else:
      emit(l2_req, req)
      emit(l1_state, state)

  @net.jit("l2_forward")
  def l2_forward():
    req = take(l2_req)
    emit(l2_resp, req, delay=L2_DELAY_NS)

  @net.jit("l1_fill")
  def l1_fill():
    resp = take(l2_resp)
    state = take(l1_state)
    line_idx = token_get(resp, "line")
    updated = token_set(state, ("line", line_idx), 1)
    emit(l1_resp, resp, delay=L1_HIT_DELAY_NS)
    emit(l1_state, updated)

  return net.build()


DEFAULT_TRACE = [0x0, 0x10, 0x20, 0x30, 0x0, 0x10]


if __name__ == "__main__":
  print(build_cache_example(DEFAULT_TRACE))
