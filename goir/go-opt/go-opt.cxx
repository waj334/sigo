// clang-format off
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Go/IR/GoDialect.h"
#include "Go/Transforms/Passes.h"
// clang-format on

int main(int argc, char** argv)
{
  mlir::registerAllPasses();
  mlir::registerPass(mlir::go::createLowerToCorePass);
  mlir::registerPass(mlir::go::createGlobalConstantsPass);
  mlir::registerPass(mlir::go::createGlobalInitializerPass);
  mlir::registerPass(mlir::go::createLowerToLLVMPass);
  mlir::registerPass(mlir::go::createHeapEscapePass);

  mlir::DialectRegistry registry;
  registry.insert<
    mlir::go::GoDialect,
    mlir::DLTIDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::complex::ComplexDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect>();

  auto result = mlir::MlirOptMain(argc, argv, "Go optimizer driver\n", registry);

  return mlir::asMainReturnCode(result);
}