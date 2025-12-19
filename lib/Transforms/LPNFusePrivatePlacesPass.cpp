#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::lpn {
namespace {

struct PlaceUsage {
  SmallVector<EmitOp> producers;
  SmallVector<TakeOp> consumers;
  bool escapes = false;
};

struct LPNFusePrivatePlacesPass
    : PassWrapper<LPNFusePrivatePlacesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNFusePrivatePlacesPass)

  StringRef getArgument() const final { return "lpn-fuse-private-places"; }
  StringRef getDescription() const final {
    return "Mark single-producer/single-consumer places as fusion candidates.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DenseMap<StringAttr, PlaceOp> places;
    module.walk([&](PlaceOp place) {
      places[place.getSymNameAttr()] = place;
    });

    DenseMap<StringAttr, PlaceUsage> usage;
    module.walk([&](PlaceRefOp ref) {
      StringAttr sym = ref.getPlaceAttr().getAttr();
      auto &info = usage[sym];
      for (Operation *user : ref.getResult().getUsers()) {
        if (auto emit = dyn_cast<EmitOp>(user)) {
          info.producers.push_back(emit);
        } else if (auto take = dyn_cast<TakeOp>(user)) {
          info.consumers.push_back(take);
        } else {
          info.escapes = true;
        }
      }
    });

    // Treat any place referenced via lpn.place_list as visible to arbitrary
    // users because the handles can be pulled out dynamically.
    module.walk([&](PlaceListOp list) {
      for (Attribute attr : list.getPlacesAttr()) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(attr);
        if (!ref)
          continue;
        usage[ref.getAttr()].escapes = true;
      }
    });

    Builder builder(module.getContext());
    for (auto &entry : usage) {
      auto it = places.find(entry.first);
      if (it == places.end())
        continue;

      PlaceUsage &info = entry.second;
      PlaceOp place = it->second;
      if (place.getObservableAttr())
        continue;
      if (info.escapes || info.producers.size() != 1 ||
          info.consumers.size() != 1)
        continue;

      place->setAttr("lpn.fuse_candidate", builder.getUnitAttr());
      auto prodTrans =
          info.producers.front()->getParentOfType<TransitionOp>();
      auto consTrans =
          info.consumers.front()->getParentOfType<TransitionOp>();
      SmallVector<NamedAttribute> plan;
      if (prodTrans) {
        plan.push_back(builder.getNamedAttr(
            "producer", FlatSymbolRefAttr::get(prodTrans.getSymNameAttr())));
      }
      if (consTrans) {
        plan.push_back(builder.getNamedAttr(
            "consumer", FlatSymbolRefAttr::get(consTrans.getSymNameAttr())));
      }
      if (!plan.empty())
        place->setAttr("lpn.fuse_plan", builder.getDictionaryAttr(plan));
      place.emitRemark()
          << "single-producer/single-consumer place; consider fusing transitions "
          << (prodTrans ? prodTrans.getSymName() : "<unknown>") << " -> "
          << (consTrans ? consTrans.getSymName() : "<unknown>");
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createLPNFusePrivatePlacesPass() {
  return std::make_unique<LPNFusePrivatePlacesPass>();
}

}  // namespace mlir::lpn
