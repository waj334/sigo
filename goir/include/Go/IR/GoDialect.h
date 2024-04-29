#ifndef GO_GODIALECT_H
#define GO_GODIALECT_H

#include <mlir/IR/Dialect.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/IR/DataLayout.h>

#include "Go/IR/GoEnums.h"
#include "Go/IR/GoAttrs.h"

#include "Go/IR/GoOpsDialect.h.inc"

#endif // GO_GODIALECT_H