#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::go {
std::unique_ptr<Pass> createGlobalInitializerPass();
} // namespace mlir::go
