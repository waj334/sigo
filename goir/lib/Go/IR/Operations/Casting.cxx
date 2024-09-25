#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"
#include "Go/Util.h"

namespace mlir::go
{
::mlir::LogicalResult BitcastOp::verify()
{
  return success();

  // Verify that primitive types are only cast between their MLIR and runtime representations and
  // that pointers are only cast to and from unsafe.Pointer.
  if (failed(llvm::TypeSwitch<mlir::Type, LogicalResult>(this->getValue().getType())
               .Case(
                 [&](ChanType T)
                 {
                   if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
                   {
                     return success(resultType.getName() == "runtime._channel");
                   }
                   return failure();
                 })
               .Case(
                 [&](ComplexType T)
                 {
                   // Allow bitcast from one complex type to another.
                   return success(go::isa<mlir::ComplexType>(this->getType()));
                 })
               .Case(
                 [&](InterfaceType T)
                 {
                   if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
                   {
                     return success(resultType.getName() == "runtime._interface");
                   }
                   return failure();
                 })
               .Case(
                 [&](MapType T)
                 {
                   if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
                   {
                     return success(resultType.getName() == "runtime._map");
                   }
                   return failure();
                 })
               .Case(
                 [&](SliceType T)
                 {
                   if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
                   {
                     return success(resultType.getName() == "runtime._slice");
                   }
                   return failure();
                 })
               .Case(
                 [&](StringType T)
                 {
                   if (auto resultType = mlir::dyn_cast<NamedType>(this->getType()))
                   {
                     return success(resultType.getName() == "runtime._string");
                   }
                   return failure();
                 })
               .Case(
                 [&](NamedType T)
                 {
                   bool isSuccess =
                     llvm::StringSwitch<bool>(T.getName().getValue())
                       .Case("runtime._channel", go::isa<ChanType>(this->getType()))
                       .Case("runtime._interface", go::isa<InterfaceType>(this->getType()))
                       .Case("runtime._map", go::isa<MapType>(this->getType()))
                       .Case("runtime._slice", go::isa<SliceType>(this->getType()))
                       .Case("runtime._string", go::isa<StringType>(this->getType()))
                       .Default(false);

                   if (!isSuccess)
                   {
                     {
                       auto fromType = go::dyn_cast<IntegerType>(T);
                       auto toType = go::dyn_cast<IntegerType>(this->getType());
                       if (fromType && toType)
                       {
                         return success(fromType.getWidth() == toType.getWidth());
                       }
                     }
                     {
                       auto fromType = go::dyn_cast<FloatType>(T);
                       auto toType = go::dyn_cast<FloatType>(this->getType());
                       if (fromType && toType)
                       {
                         return success(fromType.getWidth() == toType.getWidth());
                       }
                     }
                     {
                       auto fromType = go::dyn_cast<ComplexType>(T);
                       auto toType = go::dyn_cast<ComplexType>(this->getType());
                       if (fromType && toType)
                       {
                         auto fromFType = mlir::cast<FloatType>(fromType.getElementType());
                         auto toFType = mlir::cast<FloatType>(toType.getElementType());
                         return success(fromFType.getWidth() == toFType.getWidth());
                       }
                     }

                     // The conversion is valid if both types have the same underlying type.
                     return success(baseType(this->getType()) == baseType(T));
                   }

                   return success(isSuccess);
                 })
               .Case(
                 [&](PointerType T)
                 {
                   if (auto resultType = mlir::dyn_cast<PointerType>(this->getType()))
                   {
                     if (T.getElementType().has_value())
                     {
                       if (!resultType.getElementType().has_value())
                       {
                         // *T -> unsafe.Pointer.
                         return success(true);
                       }
                       else if (baseType(*T.getElementType()) == this->getType())
                       {
                         // Alias -> underlying type.
                         return success(true);
                       }
                     }
                     // unsafe.Pointer -> *T is always acceptable.
                     return success();
                   }
                   else if (!T.getElementType().has_value())
                   {
                     if (go::isa<FunctionType>(this->getType()))
                     {
                       // unsafe.Pointer -> func
                       return success();
                     }
                   }
                   return failure();
                 })
               .Case(
                 [&](IntegerType T)
                 {
                   if (auto resultType = mlir::dyn_cast<IntegerType>(baseType(this->getType())))
                   {
                     return success(T.getWidth() == resultType.getWidth());
                   }
                   return failure();
                 })
               .Default(
                 [&](mlir::Type T)
                 {
                   // The conversion is valid if both types have the same underlying type.
                   return success(baseType(this->getType()) == baseType(T));
                 })))
  {
    return this->emitOpError() << "invalid cast from " << this->getValue().getType() << " to "
                               << this->getType();
  }
  return success();
}

::mlir::LogicalResult FunctionToPointerOp::verify()
{
  // The resulting pointer must be opaque
  auto resultType = cast<PointerType>(this->getResult().getType());
  if (resultType.getElementType())
  {
    return emitOpError() << "the resulting pointer must be opaque";
  }
  return success();
}

::mlir::LogicalResult PointerToFunctionOp::verify()
{
  // The pointer operand must be opaque
  auto pointerType = cast<PointerType>(this->getValue().getType());
  if (pointerType.getElementType())
  {
    return emitOpError() << "the pointer operand must be opaque";
  }
  return success();
}

::mlir::LogicalResult ChangeInterfaceOp::verify()
{
  const auto sourceType = go::dyn_cast<InterfaceType>(this->getValue().getType());
  if (!sourceType)
  {
    return this->emitOpError() << "the input value MUST be an interface type";
  }

  const auto resultType = go::dyn_cast<InterfaceType>(this->getType());
  if (!resultType)
  {
    return this->emitOpError() << "the result type MUST be an interface type";
  }

  const auto sourceMethods = sourceType.getMethods();
  const auto resultMethods = resultType.getMethods();

  // Check the interface methods for compatibility.
  for (auto& method : resultMethods)
  {
    const auto sourceMethod = sourceMethods.find(method.first);
    if (sourceMethod == sourceMethods.end())
    {
      return this->emitOpError() << "the resulting interface type is missing method \""
                                 << method.first << "\" from the input interface value type";
    }

    // The signatures of the methods MUST match for the resulting interface type to be compatible
    // with the input interface.
    if (method.second != sourceMethod->second)
    {
      return this->emitOpError()
        << "the signature of method \"" << method.first
        << "\" in the resulting interface type does not match that of the input "
           "value interface type";
    }
  }

  // The resulting interface type MUST implement ALL methods of the input interface type.
  return success();
}

LogicalResult TypeAssertOp::canonicalize(TypeAssertOp op, PatternRewriter& rewriter)
{
  const auto loc = op.getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  const auto assertedType = op.getType(0);
  const auto anyType = InterfaceType::get(rewriter.getContext(), {});
  const auto boolType = BooleanType::get(rewriter.getContext());
  const auto i1Type = mlir::IntegerType::get(rewriter.getContext(), 1);
  auto originalAssertedValue = op->getOpResult(0);
  auto originalOkValue = op->getOpResult(1);

  // Get the runtime function.
  auto func = module.lookupSymbol<FuncOp>("runtime.interfaceAssert");
  const auto argTypes = func.getArgumentTypes();

  // The info type pointer is the second argument.
  const auto infoType = argTypes[1];

  // Get information about the asserted type.
  Value info = rewriter.create<TypeInfoOp>(loc, infoType, assertedType);

  // Handle 'ok' semantics.
  Value hasOk = rewriter.create<ConstantOp>(
    loc, boolType, IntegerAttr::get(i1Type, op.getNumResults() == 2 ? 1 : 0));

  // Lower to runtime call.
  const SmallVector<Type> results = { anyType, boolType };
  const SmallVector<Value> args = { op.getValue(), info, hasOk };
  auto assertCallOp = rewriter.create<RuntimeCallOp>(loc, results, "runtime.interfaceAssert", args);
  Value assertedValue = assertCallOp.getResult(0);
  Value okValue = assertCallOp.getResult(1);

  if (mlir::isa<InterfaceType>(op.getType(0)))
  {
    // The resulting interface value can be used directly.
    rewriter.replaceAllUsesWith(assertedValue, assertCallOp->getResult(0));
  }
  else
  {
    // The value must be conditionally loaded based on whether the 'ok' result.
    auto opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();

    // Create the continuation block where execution will resume.
    auto continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // The result of the assertion will be passed to the continuation block via a block parameter.
    continuationBlock->addArgument(assertedType, loc);
    rewriter.replaceAllUsesWith(originalAssertedValue, continuationBlock->getArgument(0));

    auto trueBlock = rewriter.createBlock(continuationBlock);
    auto falseBlock = rewriter.createBlock(continuationBlock);

    // Create the true block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(trueBlock);

      // Get the value from the interface.
      const SmallVector<Type> results = { PointerType::get(rewriter.getContext(), std::nullopt) };
      const SmallVector<Value> args = { assertedValue };
      auto interfaceValueCallOp =
        rewriter.create<RuntimeCallOp>(loc, results, "runtime.interfaceValue", args);
      Value underlyingValuePtr = interfaceValueCallOp->getResult(0);

      // Load the value.
      auto newAssertedValue =
        rewriter.create<LoadOp>(loc, assertedType, underlyingValuePtr, UnitAttr(), UnitAttr());

      // Branch to the continuation block while passing the asserted value.
      const SmallVector<Value> brArgs = { newAssertedValue };
      rewriter.create<BranchOp>(loc, brArgs, continuationBlock);
    }

    // Create the false block.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(falseBlock);

      // Create the zero value of the asserted type.
      Value zeroValue = rewriter.create<ZeroOp>(loc, assertedType);

      // Branch to the continuation block while passing the zero value.
      const SmallVector<Value> brArgs = { zeroValue };
      rewriter.create<BranchOp>(loc, brArgs, continuationBlock);
    }

    // Terminate the orignal block with a conditional branch.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(opBlock);

      // Conditionally branch to the true or false block.
      rewriter.create<CondBranchOp>(
        loc,
        assertCallOp.getResult(1),
        SmallVector<Value>(),
        SmallVector<Value>(),
        trueBlock,
        falseBlock);
    }
  }

  // Handle 'ok' return value if present.
  if (op.getNumResults() == 2)
  {
    rewriter.replaceAllUsesWith(originalOkValue, okValue);
  }

  // Erase the original type assert operation.
  rewriter.eraseOp(op);
  return success();
}
} // namespace mlir::go
