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

  @net.jit("split_packets")
  def split_packets():
    req = take(incoming)
    size = token_get(req, "size")
    chunk_count = divi(size, CHUNK_BYTES)
    tracker = token_set(req.clone(), "chunks", chunk_count)
    emit(bookkeeping, tracker)

    trip_count = index_cast(chunk_count, src_type="i64")

    for iv in range(trip_count):
      chunk = req.clone()
      idx_i64 = index_cast(iv, src_type="index", dst_type="i64")
      chunk = token_set(chunk, "chunk_idx", idx_i64)
      emit(intermediate, chunk)

  @net.jit("merge_packets")
  def merge_packets():
    tracker = take(bookkeeping)
    chunk_count = token_get(tracker, "chunks")
    trip_count = index_cast(chunk_count, src_type="i64")

    for _ in range(trip_count):
      take(intermediate)

    emit(outgoing, tracker)

  return net.build()


if __name__ == "__main__":
  print(build_split_merge_example())
