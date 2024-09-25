#include "Go/IR/GoDialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>

#include "Go/IR/GoAttrs.h"
#include "Go/IR/GoEnums.cpp.inc"
#include "Go/IR/GoOps.h"
#include "Go/IR/GoOpsDialect.cpp.inc"
#include "Go/IR/GoTypes.h"
#include "Go/IR/Types/Struct.h"

#define GET_TYPEDEF_CLASSES

#include "Go/IR/GoTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES

#include <stack>

#include "Go/IR/GoAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Go dialect.
//===----------------------------------------------------------------------===//

namespace
{
struct GoOpAsmDialectInterface : public mlir::OpAsmDialectInterface
{
  using OpAsmDialectInterface::OpAsmDialectInterface;

  virtual AliasResult getAlias(mlir::Type type, llvm::raw_ostream& os) const override
  {
    return llvm::TypeSwitch<::mlir::Type, AliasResult>(type)
      .Case<mlir::go::NamedType>(
        [&](auto T)
        {
          if (auto structType = mlir::go::dyn_cast<mlir::go::GoStructType>(T))
          {
            if (!structType.isLiteral())
            {
              // Check if this struct contains a recursive self-reference.
              llvm::SmallDenseSet<mlir::go::GoStructType> encountered;
              std::stack<mlir::go::GoStructType> stack;
              stack.push(structType);
              while (!stack.empty())
              {
                auto S = stack.top();
                stack.pop();
                encountered.insert(S);

                for (auto& fieldType : S.getFieldTypes())
                {
                  mlir::go::GoStructType nestedStruct;
                  if (auto memberPointer = mlir::go::dyn_cast<mlir::go::PointerType>(fieldType))
                  {
                    if (memberPointer.getElementType().has_value())
                    {
                      if (
                        auto ptrStruct = mlir::go::dyn_cast<mlir::go::GoStructType>(
                          *memberPointer.getElementType()))
                      {
                        nestedStruct = ptrStruct;
                      }
                    }
                  }
                  else if (
                    auto memberStruct = mlir::go::dyn_cast<mlir::go::GoStructType>(fieldType))
                  {
                    nestedStruct = memberStruct;
                  }

                  if (nestedStruct)
                  {
                    // Check if the struct member has the same identifier as the outer struct.
                    if (!nestedStruct.isLiteral() && nestedStruct.getId() == structType.getId())
                    {
                      // This struct is recursive.
                      return AliasResult::NoAlias;
                    }

                    // Check this struct next.
                    if (!encountered.contains(nestedStruct))
                    {
                      stack.push(nestedStruct);
                    }
                  }
                }
              }
            }
          }

          os << T.getName().str();
          return AliasResult::OverridableAlias;
        })
      .Default([](mlir::Type) { return AliasResult::NoAlias; });
  }
};
} // namespace

namespace mlir::go
{
void GoDialect::initialize()
{
  // Add dependent dialect
  this->getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
  this->getContext()->getOrLoadDialect<mlir::complex::ComplexDialect>();
  this->getContext()->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  this->getContext()->getOrLoadDialect<mlir::DLTIDialect>();
  this->getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
  this->getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  addOperations<
#define GET_OP_LIST

#include "Go/IR/GoOps.cpp.inc"

    >();

  addAttributes<
#define GET_ATTRDEF_LIST

#include "Go/IR/GoAttrDefs.cpp.inc"

    >();

  addTypes<
    mlir::go::InterfaceType,
    mlir::go::GoStructType,
#define GET_TYPEDEF_LIST

#include "Go/IR/GoTypes.cpp.inc"

    >();

  // Print type aliases.
  addInterface<GoOpAsmDialectInterface>();
}

::mlir::Type GoDialect::parseType(::mlir::DialectAsmParser& parser) const
{
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type value;

  // auto parseResult = ::generatedTypeParser(parser, &mnemonic, value);

  mlir::OptionalParseResult result =
    mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser, &mnemonic)
      .Case(
        IntegerType::getMnemonic(),
        [&](llvm::StringRef, llvm::SMLoc)
        {
          value = IntegerType::parse(parser);
          return success(!!value);
        })
      .Case(
        InterfaceType::getMnemonic(),
        [&](llvm::StringRef, llvm::SMLoc)
        {
          value = InterfaceType::parse(parser);
          return success(!!value);
        })
      .Case(
        GoStructType::getMnemonic(),
        [&](llvm::StringRef, llvm::SMLoc)
        {
          value = GoStructType::parse(parser);
          return success(!!value);
        })
      .Default([&](llvm::StringRef, llvm::SMLoc)
               { return ::generatedTypeParser(parser, &mnemonic, value); });

  if (result.has_value())
  {
    return value;
  }

  parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `" << getNamespace()
                            << "`";
  return {};

  /*
  // NOTE: No result is returned if the type does not exist in the dialect. Attempt to parse any
  non-tablegen
  //       defined types.
  if (!parseResult.has_value()) {
      parseResult = ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(
                  parser, &mnemonic)
              .Case(InterfaceType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
                  value = InterfaceType::parse(parser);
                  return success(!!value);
              })
              .Case(GoStructType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
                  value = GoStructType::parse(parser);
                  return success(!!value);
              })
              .Default([&](llvm::StringRef, llvm::SMLoc) {
                  return std::nullopt;
              });
  }

  if (parseResult.has_value())
      return value;

  parser.emitError(typeLoc) << "unknown  type `"
          << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
   */
}

/// Print a type registered to this dialect.
void GoDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& printer) const
{
  (void)::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(type)
    .Case(
      [&](const IntegerType t)
      {
        t.print(printer);
        return success();
      })
    .Case(
      [&](const InterfaceType t)
      {
        printer << ::mlir::go::InterfaceType::getMnemonic();
        t.print(printer);
        return ::mlir::success();
      })
    .Case(
      [&](const GoStructType t)
      {
        printer << GoStructType::getMnemonic();
        t.print(printer);
        return success();
      })
    .Default([&](auto t) { return ::generatedTypePrinter(t, printer); });
}

IntegerType IntegerType::get(
  mlir::MLIRContext* context,
  SignednessSemantics signedness,
  std::optional<unsigned> width)
{
  return Base::get(context, signedness, width);
}

IntegerType IntegerType::getChecked(
  ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
  ::mlir::MLIRContext* context,
  IntegerType::SignednessSemantics signedness,
  std::optional<unsigned> width)
{
  if (succeeded(verify(emitError, signedness, width)))
  {
    return get(context, signedness, width);
  }
  return {};
}

FunctionType FunctionType::get(
  mlir::MLIRContext* context,
  mlir::TypeRange inputs,
  mlir::TypeRange results,
  std::optional<mlir::Type> receiver)
{
  Type receiverType;
  if (receiver.has_value())
  {
    receiverType = receiver.value();
  }
  return Base::get(
    context, receiverType, SmallVector<Type>{ inputs }, SmallVector<Type>{ results });
}

} // namespace mlir::go
