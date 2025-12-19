# RUN: lpn-opt --lpn-retain-observables %s | FileCheck %s

module {
  lpn.net {
    lpn.place @src {observable}
    lpn.place @mid
    lpn.place @dst {observable}

    lpn.transition @produce {
      ^bb0:
        %src = lpn.place_ref @src : !lpn.place
        %mid = lpn.place_ref @mid : !lpn.place
        %tok = lpn.take %src : !lpn.place -> !lpn.token
        %zero = arith.constant 0.0 : f64
        lpn.emit %mid, %tok, %zero : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.transition @consume {
      ^bb0:
        %mid = lpn.place_ref @mid : !lpn.place
        %dst = lpn.place_ref @dst : !lpn.place
        %tok = lpn.take %mid : !lpn.place -> !lpn.token
        %zero = arith.constant 0.0 : f64
        lpn.emit %dst, %tok, %zero : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.halt
  }
}

# CHECK-LABEL: lpn.net
# CHECK: lpn.place @src {observable}
# CHECK: lpn.place @dst {observable}
# CHECK-NOT: lpn.place @mid
# CHECK: lpn.transition @src_to_dst_0
# CHECK: lpn.take %{{.*}} : !lpn.place -> !lpn.token
# CHECK: lpn.emit %{{.*}}, %{{.*}}, %{{.*}} : !lpn.place, !lpn.token, f64
