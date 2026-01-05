"""Minimal demo net showing a helper @net.func returning an LPN array.

Run with:
  python -m examples.simple_array_net --emit         # print MLIR
  python -m examples.simple_array_net --simulate     # run python simulator
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Optional

from lpnlang_mlir import NetBuilder

# Placeholders so linters understand injected names inside @net.func bodies.
create = None  # type: ignore
emit = None    # type: ignore
take = None    # type: ignore
array = None   # type: ignore
array_alloc = None # type: ignore
array_set = None # type: ignore
func_return = None  # type: ignore


def register_demo_helpers(net: NetBuilder) -> SimpleNamespace:
  helpers = SimpleNamespace()

  @net.func(args=("Place", "i64"), results=("list[Token]",))
  def take_n_tokens(place, n):
    """Take `n` tokens from `place` and return them as an LPN array."""
    # Create a dummy token for initialization
    dummy = create()
    arr = array_alloc(n, dummy)
    for i in range(n):
      tok = take(place)
      arr[i] = tok
    func_return(arr)

  helpers.take_n_tokens = take_n_tokens
  return helpers


def build_demo_net(num_tokens: int = 3, name: str = "array_demo") -> NetBuilder:
  net = NetBuilder(name)
  helpers = register_demo_helpers(net)

  seed = net.place("seed", initial_tokens=1)
  p_src = net.place("p_src")
  p_ctrl = net.place("p_ctrl")
  p_sink = net.place("p_sink", observable=True)

  @net.transition
  def bootstrap(seed=seed):
    take(seed)
    emit(p_ctrl, create().set("count", num_tokens))
    for i in range(num_tokens):
      emit(p_src, create().set("value", i + 1))

  @net.transition
  def consume():
    ctrl = take(p_ctrl)
    n = ctrl.get("count")
    arr = helpers.take_n_tokens(p_src, n)
    for i in range(n):
      emit(p_sink, arr[i])

  return net


def main(argv: Optional[list[str]] = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-tokens", type=int, default=3,
                      help="How many tokens to preload and consume")
  parser.add_argument("--simulate", action="store_true",
                      help="Run Python simulator instead of emitting MLIR")
  args = parser.parse_args(argv)

  net = build_demo_net(num_tokens=args.num_tokens)

  if args.simulate:
    sim = net.python_simulator()
    final_time = sim.run(max_time=1_000.0, debug=False)
    print(f"Simulation finished at t={final_time:.3f} ns")
    tokens = sim.place_dicts("p_sink")
    print(f"p_sink has {len(tokens)} token(s):")
    for t in tokens:
      print(f"  {t}")
  else:
    print(net.build())


if __name__ == "__main__":
  main()
