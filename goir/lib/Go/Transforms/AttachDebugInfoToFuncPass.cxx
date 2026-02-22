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

struct AttachDebugInfoToFuncPass final
  : BaseAttachDebugInfoPass<AttachDebugInfoToFuncPass, mlir::go::FuncOp>
{
  void runOnOperation() override
  {
    const auto context = &this->getContext();
    auto op = this->getOperation();
    if (op.getBody().empty())
    {
      // Skip forward declared functions.
      return;
    }

    // Functions that should generate debug information have a compile unit
    // fused with its location.
    if (const auto fusedLocWithCompileUnit =
          op->getLoc()->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
        fusedLocWithCompileUnit)
    {
      const auto compileUnitAttr = fusedLocWithCompileUnit.getMetadata();
      const auto funcNameAttr = op.getSymNameAttr();

      auto baseFuncName = funcNameAttr.strref();
      if (baseFuncName.contains("."))
      {
        // Extract the function's unqualified name.
        baseFuncName = baseFuncName.substr(baseFuncName.find_last_of(".") + 1);
      }
      auto baseFuncNameAttr = StringAttr::get(context, baseFuncName);

      const auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      const auto filePath = std::filesystem::path(loc.getFilename().str());
      const auto fileName = filePath.filename().generic_string();
      const auto fileDir = filePath.parent_path().generic_string();
      const auto fileAttr = LLVM::DIFileAttr::get(context, fileName, fileDir);
      const auto subprogramIdAttr = DistinctAttr::create(UnitAttr::get(context));
      const auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});
      const auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context,
        subprogramIdAttr,
        compileUnitAttr,
        fileAttr,
        baseFuncNameAttr,
        funcNameAttr,
        fileAttr,
        /*line=*/loc.getLine(),
        /*scopeline=*/loc.getLine(),
        LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
        subroutineTypeAttr,
        {},
        {});
      op->setLoc(FusedLoc::get(context, { op.getLoc() }, subprogramAttr));
    }
  }

  StringRef getArgument() const override { return "go-attach-debug-info-to-func-pass"; }

  StringRef getDescription() const override
  {
    return "Attach debug information to function declarations";
  }
};

std::unique_ptr<Pass> createAttachDebugInfoToFuncPass()
{
  return std::make_unique<AttachDebugInfoToFuncPass>();
}

} // namespace mlir::go