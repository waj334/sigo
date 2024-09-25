#include "Go/IR/GoOps.h"

#include <mlir/IR/OpImplementation.h>

#include "Go/IR/GoDialect.h"

namespace mlir::go
{

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static ::mlir::Type getI1SameShape(::mlir::Type type)
{
  auto i1Type = ::mlir::go::BooleanType::get(type.getContext());
  return i1Type;
}

ParseResult parseGEPIndices(
  OpAsmParser& parser,
  SmallVectorImpl<OpAsmParser::UnresolvedOperand>& dynamicIndices,
  DenseI32ArrayAttr& constIndices)
{
  SmallVector<int32_t> constantIndices;
  int32_t index = 0;
  if (parser.parseCommaSeparatedList(
        [&]() -> ParseResult
        {
          int32_t constValue;
          OptionalParseResult parsedInteger = parser.parseOptionalInteger(constValue);
          if (parsedInteger.has_value())
          {
            if (failed(parsedInteger.value()))
            {
              return failure();
            }
            constantIndices.push_back(constValue);
            return success();
          }

          constantIndices.push_back(GetElementPointerOp::kValueFlag | index++);
          return parser.parseOperand(dynamicIndices.emplace_back());
        }))
  {
    return failure();
  }

  constIndices = DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

void printGEPIndices(
  OpAsmPrinter& printer,
  GetElementPointerOp gepOp,
  OperandRange dynamicIndices,
  DenseI32ArrayAttr constIndices)
{
  llvm::interleaveComma(
    constIndices.asArrayRef(),
    printer,
    [&](int32_t value)
    {
      if (value & GetElementPointerOp::kValueFlag)
      {
        const auto index = value & GetElementPointerOp::kValueIndexMask;
        printer.printOperand(dynamicIndices[index]);
      }
      else
      {
        printer << value;
      }
    });
}

} // namespace mlir::go

#define GET_OP_CLASSES

#include "Go/IR/GoOps.cpp.inc"
