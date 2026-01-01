"""Example net where a transition takes from two observable places."""

from __future__ import annotations

from lpnlang_mlir import NetBuilder


def build_multi_take_example() -> str:
  net = NetBuilder("multi_take")

  left = net.place("left", observable=True)
  right = net.place("right", observable=True)
  merged = net.place("merged", observable=True)
  pump = net.place("pump", initial_tokens=1)

  @net.jit("seed_left")
  def seed_left():
    ctrl = take(pump)
    token = token_create()
    token = token_set(token, "payload", 7)
    emit(left, token)
    emit(pump, ctrl)

  @net.jit("seed_right")
  def seed_right():
    ctrl = take(pump)
    token = token_create()
    token = token_set(token, "payload", 13)
    emit(right, token)
    emit(pump, ctrl)

  @net.jit("merge_tokens")
  def merge_tokens():
    token_left = take(left)
    token_right = take(right)
    combined = token_set(token_left, "left_payload",
                         token_get(token_left, "payload"))
    combined = token_set(combined, "right_payload",
                         token_get(token_right, "payload"))
    emit(merged, combined)

  return net.build()


if __name__ == "__main__":
  print(build_multi_take_example())
