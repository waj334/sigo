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
#define GEN_PASS_DEF_ATTACHDEBUGINFOTOCONSTANTPASS
#include "Go/Transforms/Passes.h.inc"

namespace
{

struct AttachDebugInfoToConstantPass final
  : ::mlir::go::impl::AttachDebugInfoToConstantPassBase<AttachDebugInfoToConstantPass>
  , BaseAttachDebugInfoPass
{
  using AttachDebugInfoToConstantPassBase<
    AttachDebugInfoToConstantPass>::AttachDebugInfoToConstantPassBase;

  void runOnOperation() override
  {
    /*
        const auto context = &this->getContext();
        auto op = this->getOperation();
        const auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

        const auto runtimeTypes = RuntimeTypeLookUp(module);
        const DataLayout dataLayout(module);

        const Location loc = op->getLoc();
        const auto fusedLocWithCompileUnit =
          loc->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();

        // Globals must have an associated compile unit in order for debug information about it to
        // be emitted.
        if (!fusedLocWithCompileUnit)
          return;

        const auto elementT = op.getGlobalType();
        const auto typeAttr = this->getDITypeAttr(context, elementT, dataLayout, runtimeTypes);

        const auto alignment = dataLayout.getTypePreferredAlignment(elementT);
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
          context, diGlobalAttr, LLVM::DIExpressionAttr());
        op->setLoc(FusedLoc::get(context, { op.getLoc() }, diGlobalExprAttr));
    */
  }
};

} // namespace

} // namespace mlir::go