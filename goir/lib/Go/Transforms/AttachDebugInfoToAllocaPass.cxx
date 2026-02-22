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

struct AttachDebugInfoToAllocaPass final
  : BaseAttachDebugInfoPass<AttachDebugInfoToAllocaPass, mlir::go::AllocaOp>
{
  void runOnOperation() override
  {
    const auto context = &this->getContext();
    auto op = this->getOperation();
    const auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    const auto runtimeTypes = RuntimeTypeLookUp(module);
    const DataLayout dataLayout(module);

    const auto name = op.getVarName();

    // Get the allocated type that the debug information will be associated
    // with.
    const Type elementType = mlir::cast<TypeAttr>(op->getAttr("element")).getValue();

    // Not all allocs have debug information associated with them. Process
    // only the ones that do.
    if (name && !name->empty())
    {
      const auto locSubprogram =
        op->getParentOp()->getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>();
      if (!locSubprogram)
        return;

      const LLVM::DITypeAttr diType = getDITypeAttr(context, elementType, dataLayout, runtimeTypes);
      if (!diType)
        return;

      const auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      const auto path = std::filesystem::path(loc.getFilename().str());
      const auto diFile =
        LLVM::DIFileAttr::get(context, path.filename().string(), path.parent_path().string());

      // Apply scoping information if present.
      mlir::LLVM::DIScopeAttr scope = locSubprogram.getMetadata();
      const auto locScope = op->getLoc()->findInstanceOf<FusedLocWith<mlir::go::ScopeAttr>>();
      if (locScope)
      {
        const auto scopeAttr = locScope.getMetadata();
        mlir::LLVM::DIScopeAttr parent = locSubprogram.getMetadata();
        if (const auto parentScopeAttr = scopeAttr.getParent())
        {
          const auto start = mlir::cast<mlir::FileLineColLoc>(parentScopeAttr.getStart());
          const auto line = start.getLine();
          const auto column = start.getColumn();
          parent = mlir::LLVM::DILexicalBlockAttr::get(parent, diFile, line, column);
        }

        const auto start = mlir::cast<mlir::FileLineColLoc>(scopeAttr.getStart());
        const auto line = start.getLine();
        const auto column = start.getColumn();
        scope = mlir::LLVM::DILexicalBlockAttr::get(parent, diFile, line, column);
      }

      const auto diLocalVarAttr = LLVM::DILocalVariableAttr::get(
        scope,
        *name,
        diFile,
        loc.getLine(),                  // LINE
        0,                              // ARG,
        dataLayout.getStackAlignment(), // ALIGN
        diType,
        mlir::LLVM::DIFlags::Zero);

      // Attach the debug information to the operation. The LLVM lowering pass
      // will actually create the required operations.
      op->setLoc(FusedLoc::get(context, { op.getLoc() }, diLocalVarAttr));
    }
  }

  StringRef getArgument() const override { return "go-attach-debug-info-to-alloca-pass"; }

  StringRef getDescription() const override
  {
    return "Attach debug information to local variable declarations";
  }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<GoDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
};

std::unique_ptr<Pass> createAttachDebugInfoToAllocaPass()
{
  return std::make_unique<AttachDebugInfoToAllocaPass>();
}

} // namespace mlir::go