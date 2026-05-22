#include <filesystem>

#include <llvm/BinaryFormat/Dwarf.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/BaseAttachDebugInfoPass.h"
#include "Go/Transforms/Passes.h"
#include "Go/Transforms/TypeConverter.h"
#include "Go/Util.h"

namespace mlir::go
{
#define GEN_PASS_DEF_ATTACHDEBUGINFOTOGLOBALPASS
#include "Go/Transforms/Passes.h.inc"

namespace
{

struct AttachDebugInfoToGlobalPass final
  : ::mlir::go::impl::AttachDebugInfoToGlobalPassBase<AttachDebugInfoToGlobalPass>
  , BaseAttachDebugInfoPass
{
  using AttachDebugInfoToGlobalPassBase<
    AttachDebugInfoToGlobalPass>::AttachDebugInfoToGlobalPassBase;

  void runOnOperation() override
  {
    const auto context = &this->getContext();
    auto op = this->getOperation();
    const auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    const auto runtimeTypes = RuntimeTypeLookUp(module);
    const DataLayout dataLayout(module);

    const auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
    const auto fusedLocWithCompileUnit =
      op->getLoc()->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();

    // Globals must have an associated compiler unit in order for debug information about it to
    // be emitted.
    if (!fusedLocWithCompileUnit)
    {
      return;
    }

    const auto path = std::filesystem::path(loc.getFilename().str());
    const auto diFile =
      LLVM::DIFileAttr::get(context, path.filename().string(), path.parent_path().string());

    const auto elementT = op.getGlobalType();
    const auto typeAttr = this->getDITypeAttr(
      context, elementT, diFile, loc.getLine(), LLVM::DIScopeAttr(), dataLayout, runtimeTypes);

    const auto alignment = op.getAlignment().value_or(dataLayout.getTypePreferredAlignment(elementT));
    const auto compileUnitAttr = fusedLocWithCompileUnit.getMetadata();
    const auto linknameAttr = op.getSymNameAttr();

    std::string name = op.getSymNameAttr().str();
    if (const auto index = name.find('.'); index != std::string::npos)
    {
      name = name.substr(index + 1);
    }

    const auto nameAttr = StringAttr::get(context, name);
    const auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
    const auto filePath = std::filesystem::path(fileLoc.getFilename().str());
    const auto fileName = filePath.filename().generic_string();
    const auto fileDir = filePath.parent_path().generic_string();
    const auto fileAttr = LLVM::DIFileAttr::get(context, fileName, fileDir);
    const auto diGlobalAttr = LLVM::DIGlobalVariableAttr::get(
      context,
      compileUnitAttr,
      nameAttr,
      linknameAttr,
      fileAttr,
      fileLoc.getLine(),
      typeAttr,
      false,
      true,
      alignment);
    const auto diGlobalExprAttr = LLVM::DIGlobalVariableExpressionAttr::get(
      context,
      diGlobalAttr,
      LLVM::DIExpressionAttr::get(context, mlir::SmallVector<mlir::LLVM::DIExpressionElemAttr>()));
    op->setLoc(FusedLoc::get(context, { op.getLoc() }, diGlobalExprAttr));
  }
};

} // namespace
} // namespace mlir::go