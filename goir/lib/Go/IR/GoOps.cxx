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
  DenseI32ArrayAttr& constIndicesAttr,
  DenseBoolArrayAttr& indexFlagsAttr)
{
  SmallVector<bool> indexFlags;
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

            // The operand is a constant.
            constantIndices.push_back(constValue);
            indexFlags.push_back(false);
            return success();
          }

          // The operand is a value.
          indexFlags.push_back(true);
          return parser.parseOperand(dynamicIndices.emplace_back());
        }))
  {
    return failure();
  }

  indexFlagsAttr = DenseBoolArrayAttr::get(parser.getContext(), indexFlags);
  constIndicesAttr = DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

void printGEPIndices(
  OpAsmPrinter& printer,
  GetElementPointerOp gepOp,
  const OperandRange dynamicIndices,
  const DenseI32ArrayAttr constIndicesAttr,
  const DenseBoolArrayAttr indexFlagsAttr)
{
  size_t valuesIndex = 0;
  size_t constsIndex = 0;
  llvm::interleaveComma(
    indexFlagsAttr.asArrayRef(),
    printer,
    [&](const bool isValue)
    {
      if (isValue)
      {
        printer.printOperand(dynamicIndices[valuesIndex++]);
      } else
      {
        printer << constIndicesAttr[constsIndex++];
      }
    });
}

} // namespace mlir::go

#define GET_OP_CLASSES

#include "Go/IR/GoOps.cpp.inc"
