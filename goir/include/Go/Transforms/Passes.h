#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::go {

    std::unique_ptr<mlir::Pass> createLowerToCorePass();

    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

    std::unique_ptr<mlir::Pass> createDumpToFilePass(StringRef name, StringRef dir);

} // namespace mlir::go

