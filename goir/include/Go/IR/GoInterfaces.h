#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir::go
{

mlir::LogicalResult verifyHasCompatibleOperandsTypesTrait(mlir::Operation* op);
mlir::LogicalResult verifyHasCompatibleOperandsAndResultTypesTrait(mlir::Operation* op);

template<typename ConcreteType>
class HasCompatibleOperandsTypes
  : public ::mlir::OpTrait::TraitBase<ConcreteType, HasCompatibleOperandsTypes>
{
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation* op)
  {
    return verifyHasCompatibleOperandsTypesTrait(op);
  }
};

template<typename ConcreteType>
class HasCompatibleOperandsAndResultTypes
  : public ::mlir::OpTrait::TraitBase<ConcreteType, HasCompatibleOperandsAndResultTypes>
{
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation* op)
  {
    return verifyHasCompatibleOperandsAndResultTypesTrait(op);
  }
};

mlir::SmallVector<mlir::Type> defaultResolveOperandTypes(mlir::Operation* op);
mlir::SmallVector<mlir::Type> defaultResolveResultTypes(mlir::Operation* op);

} // namespace mlir::go

#include <Go/IR/GoAttrInterfaces.h.inc>
#include <Go/IR/GoOpInterfaces.h.inc>
#include <Go/IR/GoTypeInterfaces.h.inc>