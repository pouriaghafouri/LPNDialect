"""Demo of stripping a hidden place with multiple token emissions.

Transition T1 emits 2 tokens to hidden place B.
Transition T2 takes 1 token from B and emits to C.
"""

from lpnlang_mlir import NetBuilder

# Placeholders for DSL injection
take = None
emit = None
index_cast = None
sitofp = None
create = None

def build_strip_demo():
  net = NetBuilder("strip_demo")
  
  # A and C are observable (retained)
  A = net.place("A", initial_tokens=0, observable=True)
  C = net.place("C", observable=True)
  
  # B, D are hidden (not observable) -> candidates for stripping
  B = net.place("B") 
  D = net.place("D")

  @net.transition("Init")
  def init():
      tok = create()
      # Set count to 3
      tok = tok.set("count", 3)
      emit(A, tok)

  @net.transition("T1_LoopEmit")
  def t1():
    t = take(A)
    t2 = take(A)
    # Simulate a dynamic loop bound from token
    count_prop = t.get("count") 
    
    for i in range(count_prop):
        # Cast i to index then i64 to compare
        idx_i64 = index_cast(i, src_type="index", dst_type="i64")
        
        # If i < 1, emit to B, else emit to D
        if idx_i64 < 1:
           emit(B, t, delay=10.0)
        else:
           emit(D, t, delay=12.0)

  @net.transition("T2_Forward_B")
  def t2():
    t = take(B)
    emit(C, t, delay=5.0)

  @net.transition("T3_Forward_D")
  def t3():
    t = take(D)
    emit(C, t, delay=8.0)

  return net

if __name__ == "__main__":
  print(build_strip_demo().build())
