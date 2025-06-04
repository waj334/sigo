#include <unordered_map>

#include <mlir/Pass/Pass.h>

#include "Go/IR/GoOps.h"
#include "Go/Transforms/TypeConverter.h"

namespace mlir::go
{

struct PreprocessingPass final
  : public mlir::PassWrapper<PreprocessingPass, mlir::OperationPass<mlir::ModuleOp>>
{
  void runOnOperation() override
  {
    auto context = &this->getContext();
    auto module = getOperation();

    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<mlir::go::FuncOp>> definedFunctions;
    mlir::DenseMap<mlir::StringRef, mlir::go::FuncOp> forwardDeclarations;

    // Collect all function definitions and forward declarations.
    module.walk(
      [&](mlir::go::FuncOp funcOp)
      {
        const auto symbol = funcOp.getSymName();
        if (funcOp.getBody().empty())
        {
          if (const auto linkageAttr =
                funcOp->getAttrOfType<mlir::LLVM::LinkageAttr>("llvm.linkage");
              linkageAttr && linkageAttr.getLinkage() == mlir::LLVM::linkage::Linkage::External)
          {
            if (forwardDeclarations.contains(symbol))
            {
              // Keep at least one forward declaration.
              funcOp.erase();
            }
            else
            {
              forwardDeclarations.insert({ symbol, funcOp });
            }
          }
        }
        else
        {
          definedFunctions[symbol].push_back(funcOp);
        }
      });

    // Remove forward declared functions if their definition exists in this module to prevent
    // symbol redefinitions downstream.
    for (auto& [_, forwardDeclaration] : forwardDeclarations)
    {
      if (definedFunctions.contains(forwardDeclaration.getSymName()))
      {
        // Remove this forward declaration.
        forwardDeclaration.erase();
      }
    }

    // Remove any function definition with weak linkage that should be overridden.
    for (const auto& [_, definitions] : definedFunctions)
    {
      if (definitions.size() <= 1)
      {
        continue;
      }

      mlir::go::FuncOp strongDefinition = nullptr;
      for (const auto& definition : definitions)
      {
        const auto linkageAttr =
          mlir::dyn_cast_or_null<mlir::LLVM::LinkageAttr>(definition->getAttr("llvm.linkage"));

        if (
          !linkageAttr ||
          !llvm::is_contained(
            { mlir::LLVM::linkage::Linkage::Weak,
              mlir::LLVM::linkage::Linkage::WeakODR,
              mlir::LLVM::linkage::Linkage::ExternWeak },
            linkageAttr.getLinkage()))
        {
          strongDefinition = definition;
          break;
        }
      }

      // If no strong definition, pick the first weak one arbitrarily.
      const mlir::go::FuncOp chosenDef = strongDefinition ? strongDefinition : definitions.front();
      for (auto& def : definitions)
      {
        if (def != chosenDef)
        {
          def->erase();
        }
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createPreprocessingPass()
{
  return std::make_unique<PreprocessingPass>();
}

} // namespace mlir::go
