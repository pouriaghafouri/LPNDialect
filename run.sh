# PYTHONPATH=. python3 examples/network_net.py > /tmp/network_net.mlir
# ./build/tools/lpn-opt/lpn-opt --lpn-abstract-hidden-state --lpn-retain-observables /tmp/network_net.mlir > /tmp/network_after_pass.mlir
# PYTHONPATH=. python3 examples/cache_net.py > /tmp/cache_net.mlir
# ./build/tools/lpn-opt/lpn-opt --lpn-abstract-hidden-state --lpn-retain-observables /tmp/cache_net.mlir > /tmp/cache_after_pass.mlir
# cmake -S . -B build \
#       -DLLVM_DIR=/home/jiacma/lpn_mlir/.deps/llvm-project-main/install/lib/cmake/llvm \
#       -DMLIR_DIR=/home/jiacma/lpn_mlir/.deps/llvm-project-main/install/lib/cmake/mlir \
#       -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
PYTHONPATH=. python3 examples/cache_net.py > /tmp/cache_net.mlir

./build/tools/lpn-opt/lpn-opt --mlir-print-ir-after-all --lpn-synthesize-guards \
  --lpn-normalize-delays \
  --lpn-retain-hypergraph --lpn-strip-hidden-values \
  --lpn-resolve-choices --canonicalize --cse \
  --lpn-dataflow-simplify /tmp/cache_net.mlir > /tmp/cache_after_pass.mlir 2> /tmp/cache_pass_debug.log

PYTHONPATH=. python3 examples/split_merge_net.py > /tmp/split_merge.mlir
./build/tools/lpn-opt/lpn-opt --lpn-synthesize-guards --lpn-normalize-delays \
  --lpn-retain-hypergraph --lpn-strip-hidden-values \
  --lpn-resolve-choices --canonicalize --cse \
  --lpn-dataflow-simplify /tmp/split_merge.mlir > /tmp/split_merge_after.mlir
