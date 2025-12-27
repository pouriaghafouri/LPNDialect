"""Split/merge pipeline example that exercises scf.for in the DSL."""

from __future__ import annotations

from lpnlang_mlir import NetBuilder

CHUNK_BYTES = 16


def build_split_merge_example() -> str:
  net = NetBuilder("split_merge_net")
  incoming = net.place("in", observable=True)
  outgoing = net.place("out", observable=True)
  intermediate = net.place("chunks")
  bookkeeping = net.place("bookkeeping")

  @net.transition("split_packets")
  def _(t):
    req = t.take(incoming)
    size = t.token_get(req, "size")
    chunk_size = t.i64(CHUNK_BYTES)
    chunk_count = t.divi(size, chunk_size)
    tracker = req.clone()
    tracker = t.token_set(tracker, "chunks", chunk_count)
    t.emit(bookkeeping, tracker)

    trip_count = t.index_cast(chunk_count, src_type="i64", dst_type="index")
    zero = t.index(0)

    def emit_chunk(tb, iv):
      chunk = req.clone()
      idx_i64 = tb.index_cast(iv, src_type="index", dst_type="i64")
      chunk = tb.token_set(chunk, "chunk_idx", idx_i64)
      tb.emit(intermediate, chunk)

    t.for_range(zero, trip_count, body=emit_chunk)

  @net.transition("merge_packets")
  def _(t):
    tracker = t.take(bookkeeping)
    chunk_count = t.token_get(tracker, "chunks")
    trip_count = t.index_cast(chunk_count, src_type="i64", dst_type="index")
    zero = t.index(0)

    def drain(tb, iv):
      tb.take(intermediate)

    t.for_range(zero, trip_count, body=drain)
    t.emit(outgoing, tracker)

  return net.build()


if __name__ == "__main__":
  print(build_split_merge_example())
