"""Example net where a transition takes from two observable places."""

from __future__ import annotations

from lpnlang_mlir import NetBuilder


def build_multi_take_example() -> str:
  net = NetBuilder("multi_take")

  left = net.place("left", observable=True)
  right = net.place("right", observable=True)
  merged = net.place("merged", observable=True)
  pump = net.place("pump", initial_tokens=1)

  @net.transition("seed_left")
  def _(t):
    ctrl = t.take(pump)
    token = t.token_create()
    token = t.token_set(token, "payload", t.const_i64(7))
    t.emit(left, token)
    t.emit(pump, ctrl)

  @net.transition("seed_right")
  def _(t):
    ctrl = t.take(pump)
    token = t.token_create()
    token = t.token_set(token, "payload", t.const_i64(13))
    t.emit(right, token)
    t.emit(pump, ctrl)

  @net.transition("merge_tokens")
  def _(t):
    token_left = t.take(left)
    token_right = t.take(right)
    combined = t.token_set(token_left, "left_payload",
                           t.token_get(token_left, "payload"))
    combined = t.token_set(combined, "right_payload",
                           t.token_get(token_right, "payload"))
    t.emit(merged, combined)

  return net.build()


if __name__ == "__main__":
  print(build_multi_take_example())
