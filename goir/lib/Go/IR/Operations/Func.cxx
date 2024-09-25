#include <mlir/Interfaces/FunctionImplementation.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

namespace mlir::go
{

void FuncOp::build(
  ::mlir::OpBuilder& odsBuilder,
  ::mlir::OperationState& odsState,
  StringRef name,
  FunctionType type,
  ArrayRef<NamedAttribute> attrs)
{
  odsState.addAttribute(SymbolTable::getSymbolAttrName(), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), TypeAttr::get(type));
  odsState.attributes.append(attrs.begin(), attrs.end());
  odsState.addRegion();
}

::mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser& p, ::mlir::OperationState& result)
{
  StringAttr symbolNameAttr;
  SmallVector<OpAsmParser::Argument> args;
  std::optional<Type> receiverType;
  SmallVector<Type> inputTypes;
  SmallVector<Type> resultTypes;

  // Functions can either start with a symbol name or begin with a receiver.
  if (failed(p.parseOptionalSymbolName(symbolNameAttr)))
  {
    // Parse the receiver.
    if (p.parseLSquare())
    {
      return p.emitError(p.getCurrentLocation()) << "expected `[`";
    }

    OpAsmParser::Argument receiverArg;
    auto typeParseResult = p.parseOptionalArgument(receiverArg, true);
    if (typeParseResult.has_value())
    {
      if (failed(*typeParseResult))
      {
        return p.emitError(p.getCurrentLocation()) << "expected receiver argument";
      }
    }
    else
    {
      if (p.parseType(receiverArg.type))
      {
        return p.emitError(p.getCurrentLocation()) << "expected receiver type";
      }
    }

    args.push_back(receiverArg);
    receiverType = receiverArg.type;

    if (p.parseRSquare())
    {
      return p.emitError(p.getCurrentLocation()) << "expected `]`";
    }

    if (p.parseSymbolName(symbolNameAttr))
    {
      return p.emitError(p.getCurrentLocation()) << "expected symbol name";
    }
  }

  // Parse function arguments.
  if (failed(p.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult
        {
          OpAsmParser::Argument arg;
          auto result = p.parseOptionalArgument(arg, true, false);
          if (result.has_value())
          {
            if (failed(*result))
            {
              // Argument is malformed.
              return p.emitError(p.getCurrentLocation())
                << "failure while parsing arg #" << args.size();
            }
            args.push_back(arg);
            inputTypes.push_back(arg.type);
          }
          else
          {
            OpAsmParser::Argument arg;
            if (p.parseType(arg.type))
            {
              return p.emitError(p.getCurrentLocation()) << "expected argument type";
            }
            args.push_back(arg);
          }
          return success();
        })))
  {
    return p.emitError(p.getCurrentLocation()) << "expected argument list";
  }

  // Parse result types if present.
  if (succeeded(p.parseOptionalArrow()))
  {
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
    {
      return p.emitError(p.getCurrentLocation()) << "expected result types";
    }
  }

  // Parse function attributes.
  NamedAttrList parsedAttrs;
  if (failed(p.parseOptionalAttrDictWithKeyword(parsedAttrs)))
  {
    return p.emitError(p.getCurrentLocation()) << "failure while parsing attribute dictionary";
  }
  result.attributes.append(parsedAttrs);

  // Parse the optional body.
  auto body = result.addRegion();
  auto loc = p.getCurrentLocation();

  if (auto parseResult = p.parseOptionalRegion(*body, args, false); parseResult.has_value())
  {
    if (failed(*parseResult))
    {
      return p.emitError(loc) << "failed to parse region";
    }

    // Function body must NOT be empty.
    if (body->empty())
    {
      return p.emitError(loc) << "expected non-empty function body";
    }
  }

  // Create the resulting function type.
  const auto fnT = FunctionType::get(p.getContext(), inputTypes, resultTypes, receiverType);

  // Construct the operation.
  result.addAttribute(getFunctionTypeAttrName(result.name), mlir::TypeAttr::get(fnT));
  result.addAttribute(SymbolTable::getSymbolAttrName(), symbolNameAttr);
  result.addAttributes(parsedAttrs);
  return success();
}

void FuncOp::print(::mlir::OpAsmPrinter& p)
{
  const auto symbolName = this->getSymName();
  const auto fnT = this->getFunctionType();
  const auto resultTypes = fnT.getResults();

  Region& body = this->getRegion();
  const bool isExternal = body.empty();
  unsigned argIndex = 0;

  // Print optional receiver.
  if (fnT.getReceiver())
  {
    p << "[";

    if (isExternal)
    {
      p << this->getBody().getArgument(0);
      argIndex++;
    }
    else
    {
      p << fnT.getReceiver();
    }

    p << "] ";
  }

  p << symbolName << "(";
  for (; argIndex < body.getNumArguments(); argIndex++)
  {
    if (!isExternal)
    {
      p.printRegionArgument(body.getArgument(argIndex));
    }
    else
    {
      p << fnT.getInput(argIndex);
    }
    if (argIndex != body.getNumArguments() - 1)
    {
      p << ", ";
    }
  }
  p << ")";

  // Print optional result types.
  if (!resultTypes.empty())
  {
    p << " -> ";
    auto wrapped =
      !llvm::hasSingleElement(resultTypes) || llvm::isa<FunctionType>((*resultTypes.begin()));
    if (wrapped)
      p << '(';
    llvm::interleaveComma(resultTypes, p);
    if (wrapped)
      p << ')';
  }

  // Print the optional function body.
  if (!body.empty())
  {
    p << ' ';
    p.printRegion(
      body,
      /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/true);
  }
}

LogicalResult FuncOp::verify()
{
  return success();
}

} // namespace mlir::go