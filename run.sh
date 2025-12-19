# PYTHONPATH=. python3 examples/network_net.py > /tmp/network_net.mlir
# ./build/tools/lpn-opt/lpn-opt --lpn-abstract-hidden-state --lpn-retain-observables /tmp/network_net.mlir > /tmp/network_after_pass.mlir
# PYTHONPATH=. python3 examples/cache_net.py > /tmp/cache_net.mlir
# ./build/tools/lpn-opt/lpn-opt --lpn-abstract-hidden-state --lpn-retain-observables /tmp/cache_net.mlir > /tmp/cache_after_pass.mlir
cmake --build build
PYTHONPATH=. python3 examples/cache_net.py > /tmp/cache_net.mlir
./build/tools/lpn-opt/lpn-opt --lpn-normalize-delays --lpn-abstract-hidden-state --lpn-retain-observables --canonicalize --cse /tmp/cache_net.mlir > /tmp/cache_after_pass.mlir
