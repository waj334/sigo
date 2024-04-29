#include <mlir/IR/OpImplementation.h>

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoOps.h"

#include "Go/Util.h"
#include "Go/IR/GoTypes.h"

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static ::mlir::Type getI1SameShape(::mlir::Type type) {
    auto i1Type = ::mlir::IntegerType::get(type.getContext(), 1);
    return i1Type;
}

#define GET_OP_CLASSES

#include "Go/IR/GoOps.cpp.inc"
