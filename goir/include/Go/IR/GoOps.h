#pragma once

#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoInterfaces.h"
#include "Go/IR/GoTypes.h"
#include "Go/Util.h"

#define GET_OP_CLASSES

#include "Go/IR/GoOps.h.inc"

namespace mlir::go
{
template<typename T>
MlirOperation _createBinOp(
  MlirContext context,
  MlirType resultType,
  MlirValue x,
  MlirValue y,
  MlirLocation location)
{
  auto _context = unwrap(context);
  auto _resultType = unwrap(resultType);
  auto _x = unwrap(x);
  auto _y = unwrap(y);
  auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = builder.create<T>(_location, _resultType, _x, _y);
  return wrap(op);
}

template<typename T>
MlirOperation _createUnOp(MlirContext context, MlirValue x, MlirLocation location)
{
  auto _context = unwrap(context);
  auto _x = unwrap(x);
  auto _location = unwrap(location);

  mlir::OpBuilder builder(_context);
  mlir::Operation* op = builder.create<T>(_location, _x.getType(), _x);
  return wrap(op);
}

ParseResult parseGEPIndices(
  OpAsmParser& parser,
  SmallVectorImpl<OpAsmParser::UnresolvedOperand>& dynamicIndices,
  DenseI32ArrayAttr& constIndices);

void printGEPIndices(
  OpAsmPrinter& printer,
  GetElementPointerOp gepOp,
  OperandRange dynamicIndices,
  DenseI32ArrayAttr constIndices);
} // namespace mlir::go

inline bool hasUntypedTypes(mlir::Operation* op)
{
  for (const auto& t : op->getOperandTypes())
  {
    if (mlir::isa<mlir::go::UntypedType>(t))
    {
      return true;
    }
  }

  for (const auto& t : op->getResultTypes())
  {
    if (mlir::isa<mlir::go::UntypedType>(t))
    {
      return true;
    }
  }

  return false;
}

template<typename OpT>
mlir::SmallVector<mlir::Type> binOpResolveTypes(OpT* op, const size_t count)
{
  auto lhsType = op->getLhs().getType();
  auto rhsType = op->getRhs().getType();
  auto resultType = op->getResult().getType();

  mlir::Type resolvedType;

  if (!mlir::isa<mlir::go::UntypedType>(lhsType))
  {
    resolvedType = lhsType;
  }
  else if (!mlir::isa<mlir::go::UntypedType>(rhsType))
  {
    resolvedType = rhsType;
  }
  else if (!mlir::isa<mlir::go::UntypedType>(resultType))
  {
    resolvedType = resultType;
  }
  else
  {
    auto untyped = mlir::cast<mlir::go::UntypedType>(lhsType);
    resolvedType = untyped.getDefaultType();
  }

  return mlir::SmallVector<mlir::Type>(count, resolvedType);
}

template<typename OpT>
mlir::SmallVector<mlir::Type> unOpResolveTypes(OpT* op)
{
  const auto operandType = op->getOperand().getType();
  const auto resultType = op->getResult().getType();
  mlir::Type resolvedType;

  if (!mlir::isa<mlir::go::UntypedType>(operandType))
  {
    resolvedType = operandType;
  }
  else if (!mlir::isa<mlir::go::UntypedType>(resultType))
  {
    resolvedType = resultType;
  }
  else
  {
    const auto untypedType = mlir::cast<mlir::go::UntypedType>(op->getResult().getType());
    resolvedType = untypedType.getDefaultType();
  }
  return { resolvedType };
}

inline mlir::SmallVector<mlir::Type> callOpResolveOperandTypes(mlir::go::CallOp* op)
{
  mlir::SmallVector<mlir::Type> result;
  mlir::go::FunctionType fnT;
  if (const auto symbolName = op->getCallee())
  {
    auto moduleOp = op->getOperation()->getParentOfType<mlir::ModuleOp>();
    auto fnOp = moduleOp.lookupSymbol<mlir::go::FuncOp>(*symbolName);
    fnT = fnOp.getFunctionType();
  }
  else
  {
    result.push_back({});
    fnT = mlir::cast<mlir::go::FunctionType>(*op->getSignature());
  }

  if (const auto recvT = fnT.getReceiver())
  {
    result.push_back(recvT);
  }
  llvm::append_range(result, fnT.getInputs());
  return result;
}

inline mlir::SmallVector<mlir::Type> callOpResolveResultTypes(mlir::go::CallOp* op)
{
  mlir::go::FunctionType fnT;
  if (const auto symbolName = op->getCallee())
  {
    auto moduleOp = op->getOperation()->getParentOfType<mlir::ModuleOp>();
    auto fnOp = moduleOp.lookupSymbol<mlir::go::FuncOp>(*symbolName);
    fnT = fnOp.getFunctionType();
  }
  else
  {
    fnT = mlir::cast<mlir::go::FunctionType>(*op->getSignature());
  }
  return mlir::SmallVector<mlir::Type>(fnT.getResults());
}

template<typename OpT>
mlir::SmallVector<mlir::Type> specialCallOpResolveOperandTypes(OpT* op)
{
  mlir::SmallVector<mlir::Type> result;
  mlir::go::FunctionType fnT;
  if (const auto symbolName = op->getSymName())
  {
    auto moduleOp = op->getOperation()->template getParentOfType<mlir::ModuleOp>();
    auto fnOp = moduleOp.template lookupSymbol<mlir::go::FuncOp>(*symbolName);
    fnT = fnOp.getFunctionType();
  }
  else if (op->getCalleeValue())
  {
    result.push_back({});
    fnT = mlir::cast<mlir::go::FunctionType>(*op->getSignature());
  }
  else if (const auto iface = op->getIfaceValue())
  {
    result.push_back({});
    const auto ifaceType = mlir::go::dyn_cast<mlir::go::InterfaceType>(iface.getType());
    fnT = mlir::cast<mlir::go::FunctionType>(
      ifaceType.getMethods().find(op->getMethodName()->str())->second);
  }
  else
  {
    assert(false && "unreachable");
  }

  if (const auto recvT = fnT.getReceiver())
  {
    result.push_back(recvT);
  }

  llvm::append_range(result, fnT.getInputs());

  return result;
}

template<typename OpT>
mlir::SmallVector<mlir::Type> specialCallOpResolveResultTypes(OpT* op)
{
  mlir::go::FunctionType fnT;
  if (const auto symbolName = op->getSymName())
  {
    auto moduleOp = op->getOperation()->template getParentOfType<mlir::ModuleOp>();
    auto fnOp = moduleOp.template lookupSymbol<mlir::go::FuncOp>(*symbolName);
    fnT = fnOp.getFunctionType();
  }
  else if (op->getCalleeValue())
  {
    fnT = mlir::cast<mlir::go::FunctionType>(*op->getSignature());
  }
  else if (const auto iface = op->getIfaceValue())
  {
    const auto ifaceType = mlir::go::dyn_cast<mlir::go::InterfaceType>(iface.getType());
    fnT = mlir::cast<mlir::go::FunctionType>(
      ifaceType.getMethods().find(op->getMethodName()->str())->second);
  }
  else
  {
    assert(false && "unreachable");
  }

  mlir::SmallVector<mlir::Type> result(fnT.getInputs());
  if (const auto recvT = fnT.getReceiver())
  {
    result.insert(result.begin(), recvT);
  }
  return result;
}

inline mlir::SmallVector<mlir::Type> returnOpResolveOperandTypes(mlir::go::ReturnOp* op)
{
  auto parentFuncOp = op->getOperation()->template getParentOfType<mlir::go::FuncOp>();
  return mlir::SmallVector<mlir::Type>(parentFuncOp.getFunctionType().getResults());
}

inline mlir::SmallVector<mlir::Type> interfaceCallOpResolveOperandTypes(
  mlir::go::InterfaceCallOp* op)
{
  mlir::SmallVector<mlir::Type> result = { {} };
  const auto ifaceType = mlir::go::dyn_cast<mlir::go::InterfaceType>(op->getIface().getType());
  const auto fnT =
    mlir::cast<mlir::go::FunctionType>(ifaceType.getMethods().find(op->getCallee().str())->second);
  llvm::append_range(result, fnT.getInputs());
  return result;
}

inline mlir::SmallVector<mlir::Type> interfaceCallOpResolveResultTypes(
  mlir::go::InterfaceCallOp* op)
{
  const auto ifaceType = mlir::go::dyn_cast<mlir::go::InterfaceType>(op->getIface().getType());
  const auto fnT =
    mlir::cast<mlir::go::FunctionType>(ifaceType.getMethods().find(op->getCallee().str())->second);
  return mlir::SmallVector<mlir::Type>(fnT.getResults());
}

inline mlir::SmallVector<mlir::Type> storeOpResolveOperandTypes(mlir::go::StoreOp* op)
{
  mlir::SmallVector<mlir::Type> result(op->getOperation()->getOperandTypes());
  const auto addrType = mlir::cast<mlir::go::PointerType>(op->getAddr().getType());
  if (mlir::isa<mlir::go::UntypedType>(result[0]))
  {
    result[0] = (*addrType.getElementType());
  }
  return result;
}

inline mlir::SmallVector<mlir::Type> insertOpResolveOperandTypes(mlir::go::InsertOp* op)
{
  mlir::SmallVector<mlir::Type> result(op->getOperation()->getOperandTypes());
  const auto aggregateType = mlir::go::baseType(op->getAggregate().getType());
  const auto index = op->getIndex();
  if (mlir::isa<mlir::go::UntypedType>(result[0]))
  {
    result[0] =
      mlir::TypeSwitch<mlir::Type, mlir::Type>(aggregateType)
        .Case([&](const mlir::go::GoStructType structType)
              { return structType.getFieldType(index); })
        .Case([&](const mlir::go::ArrayType arrayType) { return arrayType.getElementType(); })
        .Default(
          [&](auto) -> ::mlir::Type
          {
            assert(false && "unreachable");
            return {};
          });
  }
  return result;
}

inline mlir::SmallVector<mlir::Type> yieldOpResolveOperandTypes(mlir::go::YieldOp* op)
{
  mlir::SmallVector<mlir::Type> result(op->getOperation()->getOperandTypes());
  if (auto parentOp = op->getOperation()->getParentOfType<mlir::go::GlobalOp>())
  {
    if (const auto initializerBlock = parentOp.getInitializerBlock(); initializerBlock && !initializerBlock->empty())
    {
      if (mlir::isa<mlir::go::UntypedType>(result[0]))
      {
        const auto globalType = parentOp.getGlobalType();
        result[0] = globalType;
      }
    }
  }
  return result;
}

inline mlir::SmallVector<mlir::Type> yieldOpResolveResultTypes(mlir::go::YieldOp* op)
{
  mlir::SmallVector<mlir::Type> result(op->getOperation()->getResultTypes());
  if (auto parentOp = op->getOperation()->getParentOfType<mlir::go::GlobalOp>())
  {
    if (const auto initializerBlock = parentOp.getInitializerBlock(); initializerBlock && !initializerBlock->empty())
    {
      if (mlir::isa<mlir::go::UntypedType>(result[0]))
      {
        const auto globalType = parentOp.getGlobalType();
        result[0] = globalType;
      }
    }
  }
  return result;
}
