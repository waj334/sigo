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
