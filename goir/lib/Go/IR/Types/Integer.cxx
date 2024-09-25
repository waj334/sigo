#include <Go/IR/GoTypes.h>
#include <mlir/IR/OpImplementation.h>

namespace mlir::go
{

// NOTE: the get methods are in GoDialect.cxx since those can't compile here without including the
//       implementation.

mlir::Type IntegerType::parse(::mlir::AsmParser& p)
{
  StringRef mnemonic;
  std::optional<unsigned> width;
  SignednessSemantics semantics;

  if (p.parseKeyword(&mnemonic))
  {
    p.emitError(p.getNameLoc(), "invalid integer type");
    return {};
  }

  unsigned widthValue;
  OptionalParseResult widthParseResult = p.parseOptionalInteger(widthValue);
  if (widthParseResult.has_value())
  {
    if (failed(*widthParseResult))
    {
      return {};
    }
    width = widthValue;
  }

  if (mnemonic == "i")
  {
    semantics = SignednessSemantics::Signed;
  }
  else if (mnemonic == "ui")
  {
    semantics = SignednessSemantics::Unsigned;
  }
  else if (mnemonic == "uiptr")
  {
    semantics = SignednessSemantics::Uintptr;
  }
  else
  {
    p.emitError(p.getNameLoc(), "unknown integer type mnemonic");
    return {};
  }

  return get(p.getContext(), semantics, width);
}

void IntegerType::print(::mlir::AsmPrinter& p) const
{
  switch (this->getSignedness())
  {
    case SignednessSemantics::Signed:
      p << "i";
      break;
    case SignednessSemantics::Unsigned:
      p << "ui";
      break;
    case SignednessSemantics::Uintptr:
      p << "uiptr";
      return;
  }

  if (auto width = this->getWidth())
  {
    p << *width;
  }
}

LogicalResult IntegerType::verify(
  ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
  SignednessSemantics signedness,
  std::optional<unsigned> width)
{
  if (signedness == SignednessSemantics::Uintptr && width.has_value())
  {
    return emitError() << "Uintptr must not specify any width";
  }
  return success();
}

} // namespace mlir::go