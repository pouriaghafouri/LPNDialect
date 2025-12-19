# RUN: lpn-opt --lpn-normalize-delays %s | FileCheck %s

module {
  lpn.net {
    lpn.place @src
    lpn.place @dst

    lpn.transition @producer {
      ^bb0:
        %src = lpn.place_ref @src : !lpn.place
        %dst = lpn.place_ref @dst : !lpn.place
        %token = lpn.token.create() : () -> !lpn.token
        %delay = arith.constant 5.500000 : f64
        lpn.emit %src, %token, %delay : !lpn.place, !lpn.token, f64
        %clone = lpn.token.clone %token : !lpn.token -> !lpn.token
        %z = arith.constant 0.0 : f64
        lpn.emit %dst, %clone, %z : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.halt
  }
}

# CHECK-LABEL: lpn.transition @producer
# CHECK: %[[SRC:.*]] = lpn.place_ref @src : !lpn.place
# CHECK: %[[ZERO_NEW:.*]] = arith.constant 0.0 : f64
# CHECK: lpn.emit %[[SRC]], %{{.*}}, %[[ZERO_NEW]] : !lpn.place, !lpn.token, f64
# CHECK: %[[ZERO_OLD:.*]] = arith.constant 0.0 : f64
# CHECK: lpn.emit %{{.*}}, %{{.*}}, %[[ZERO_OLD]] : !lpn.place, !lpn.token, f64
