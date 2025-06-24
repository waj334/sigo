#include <mlir/IR/OpImplementation.h>

#include <Go/IR/GoTypes.h>

namespace mlir::go
{

mlir::Type UntypedType::parse(::mlir::AsmParser& p)
{
  if (p.parseLess())
  {
    p.emitError(p.getNameLoc()) << "expected `<`";
  }

  mlir::StringRef strBaseType;
  if (p.parseKeyword(&strBaseType))
  {
    p.emitError(p.getNameLoc()) << "expected underlying type";
  }

  if (p.parseGreater())
  {
    p.emitError(p.getNameLoc()) << "expected `>`";
  }

  mlir::go::UntypedBasicKind baseTypeValue = {};
  if (strBaseType == "complex")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Complex;
  }
  else if (strBaseType == "integer")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Integer;
  }
  else if (strBaseType == "float")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Float;
  }
  else if (strBaseType == "bool")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Boolean;
  }
  else if (strBaseType == "nil")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Nil;
  }
  else if (strBaseType == "rune")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::Rune;
  }
  else if (strBaseType == "string")
  {
    baseTypeValue = mlir::go::UntypedBasicKind::String;
  }
  else
  {
    p.emitError(p.getNameLoc()) << "invalid underlying type \"" << strBaseType << "\"";
  }

  return get(p.getContext(), baseTypeValue);
}

void UntypedType::print(::mlir::AsmPrinter& p) const
{
  p << "<";

  switch (this->getBasicKind().getValue())
  {
    case mlir::go::UntypedBasicKind::Complex:
      p << "complex";
      break;
    case mlir::go::UntypedBasicKind::Boolean:
      p << "bool";
      break;
    case mlir::go::UntypedBasicKind::Float:
      p << "float";
      break;
    case mlir::go::UntypedBasicKind::Integer:
      p << "integer";
      break;
    case mlir::go::UntypedBasicKind::Nil:
      p << "nil";
      break;
    case mlir::go::UntypedBasicKind::Rune:
      p << "rune";
      break;
    case mlir::go::UntypedBasicKind::String:
      p << "string";
      break;
  }

  p << ">";
}

} // namespace mlir::go