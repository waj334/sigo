//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "Go/IR/GoDialect.h"

#include <mlir/InitAllTranslations.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

int main(int argc, char **argv) {
  mlir::registerAllTranslations();



  mlir::TranslateFromMLIRRegistration withdescription(
      "option", "different from option",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        return mlir::LogicalResult::success();
      },
      [](mlir::DialectRegistry &a) {});

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
