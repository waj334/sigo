#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::go {
    std::unique_ptr<Pass> createAttachDebugInfoPass();
} // namespace mlir::go
