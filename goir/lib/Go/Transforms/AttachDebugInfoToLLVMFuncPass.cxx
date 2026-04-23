#include <filesystem>

#include <llvm/BinaryFormat/Dwarf.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "Go/Transforms/Passes.h"

namespace mlir::go
{
#define GEN_PASS_DEF_ATTACHDEBUGINFOTOLLVMFUNCPASS
#include "Go/Transforms/Passes.h.inc"

namespace
{

/// Attempt to extract a FileLineColLoc from a possibly-nested location.
static FileLineColLoc extractFileLoc(Location loc)
{
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
  {
    for (auto innerLoc : fusedLoc.getLocations())
    {
      if (auto fileLoc = extractFileLoc(innerLoc))
        return fileLoc;
    }
  }
  if (auto callerLoc = dyn_cast<CallSiteLoc>(loc))
    return extractFileLoc(callerLoc.getCaller());
  return {};
}

struct AttachDebugInfoToLLVMFuncPass final
  : impl::AttachDebugInfoToLLVMFuncPassBase<AttachDebugInfoToLLVMFuncPass>
{
  using AttachDebugInfoToLLVMFuncPassBase<
    AttachDebugInfoToLLVMFuncPass>::AttachDebugInfoToLLVMFuncPassBase;

  void runOnOperation() override
  {
    auto op = getOperation();
    auto* context = &getContext();

    // Skip forward-declared functions.
    if (op.getBody().empty())
      return;

    // Skip functions that already have debug info (e.g. GoIR-derived).
    if (op.getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    // Need a FileLineColLoc to create debug info from.
    auto fileLoc = extractFileLoc(op.getLoc());
    if (!fileLoc)
      return;

    const auto filePath = std::filesystem::path(fileLoc.getFilename().str());
    const auto fileName = filePath.filename().generic_string();
    const auto fileDir = filePath.parent_path().generic_string();
    const auto fileAttr = LLVM::DIFileAttr::get(context, fileName, fileDir);

    const auto compileUnitIdAttr = DistinctAttr::create(UnitAttr::get(context));
    const auto producerAttr = StringAttr::get(context, "sigo");
    const auto compileUnitAttr = LLVM::DICompileUnitAttr::get(
      compileUnitIdAttr,
      llvm::dwarf::DW_LANG_C17,
      fileAttr,
      producerAttr,
      /*isOptimized=*/false,
      LLVM::DIEmissionKind::Full);

    const auto funcNameAttr = op.getSymNameAttr();
    const auto subprogramIdAttr = DistinctAttr::create(UnitAttr::get(context));
    const auto subroutineTypeAttr =
      LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});
    const auto subprogramAttr = LLVM::DISubprogramAttr::get(
      context,
      subprogramIdAttr,
      compileUnitAttr,
      fileAttr,
      funcNameAttr,
      funcNameAttr,
      fileAttr,
      /*line=*/fileLoc.getLine(),
      /*scopeline=*/fileLoc.getLine(),
      LLVM::DISubprogramFlags::Definition,
      subroutineTypeAttr,
      /*retainedNodes=*/{},
      /*annotations=*/{});

    op->setLoc(FusedLoc::get(context, { op.getLoc() }, subprogramAttr));
  }
};

} // namespace

} // namespace mlir::go
