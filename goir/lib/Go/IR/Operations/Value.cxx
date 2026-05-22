#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include <Go/IR/GoAttrDefs.h.inc>

namespace mlir::go
{

OpFoldResult ZeroOp::fold(FoldAdaptor adaptor)
{
  const auto context = this->getContext();
  return mlir::TypeSwitch<mlir::Type, OpFoldResult>(this->getResult().getType())
    .Case<mlir::go::BooleanType>([&](auto) { return mlir::BoolAttr::get(context, false); })
    .Case<mlir::go::IntegerType>(
      [&](auto) { return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), 0); })
    .Case<mlir::FloatType>([&](auto type) { return mlir::FloatAttr::get(type, 0); })
    .Case<mlir::ComplexType>([&](auto type)
                             { return mlir::go::ComplexNumberAttr::get(context, 0, 0); })
    .Case<mlir::go::StringType>([&](auto) { return mlir::StringAttr::get(context, ""); })
    .Default([&](auto) { return OpFoldResult{}; });
}

} // namespace mlir::go
