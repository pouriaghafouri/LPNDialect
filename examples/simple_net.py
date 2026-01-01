from lpnlang_mlir import NetBuilder


def build_example() -> str:
  net = NetBuilder("scheduler")
  p_in = net.place("P_in")
  p_out = net.place("P_out")
  ctrl = net.place("P_ctrl", initial_tokens=1)

  @net.jit("start")
  def start():
    ticket = take(ctrl)
    token = take(p_in)
    emit(p_out, token)
    emit(ctrl, ticket)

  return net.build()


if __name__ == "__main__":
  mlir_text = build_example()
  print(mlir_text)
