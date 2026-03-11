#include "Go-c/mlir/Passes.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>

#include "Go/Transforms/Passes.h"

using namespace mlir::go;
#include "Go/Transforms/Passes.capi.cpp.inc"