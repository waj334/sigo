#include <Go/Transforms/GlobalConstantsPass.h>
#include <Go/Transforms/OptimizeDefersPass.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Go/IR/GoDialect.h"
#include "Go/Transforms/AttachDebugInfoPass.h"
#include "Go/Transforms/GlobalInitializerPass.h"
#include "Go/Transforms/LowerTypeInfoToGo.h"
#include "Go/Transforms/HeapEscapePass.h"
#include "Go/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerPass(mlir::go::createAttachDebugInfoPass);
  mlir::registerPass(mlir::go::createLowerTypeInfoPass);
  mlir::registerPass(mlir::go::createLowerToCorePass);
  mlir::registerPass(mlir::go::createGlobalConstantsPass);
  mlir::registerPass(mlir::go::createGlobalInitializerPass);
  mlir::registerPass(mlir::go::createLowerToLLVMPass);
  mlir::registerPass(mlir::go::createOptimizeDefersPass);
  mlir::registerPass(mlir::go::createHeapEscapePass);

  mlir::DialectRegistry registry;
  registry.insert<mlir::go::GoDialect, mlir::DLTIDialect, mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::complex::ComplexDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();

  auto result = mlir::MlirOptMain(argc, argv, "Go optimizer driver\n", registry);

  return mlir::asMainReturnCode(result);
}