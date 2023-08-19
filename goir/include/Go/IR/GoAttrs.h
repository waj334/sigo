
#pragma once

#include "Go/IR/GoDialect.h"
#include "Go/IR/GoTypes.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

#include <optional>

#define GET_ATTRDEF_CLASSES
#include "Go/IR/GoAttrDefs.h.inc"