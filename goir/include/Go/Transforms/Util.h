#pragma once

#include "Go/IR/GoOps.h"

namespace mlir::go
{

bool isImmutableGlobal(go::GlobalOp globalOp);
bool canMaterializeAsGlobalInitializer(go::GlobalOp globalOp);
bool isConstantInitializerValue(
  mlir::Value value,
  llvm::DenseMap<mlir::Value, bool>& memo,
  llvm::DenseSet<go::GlobalOp>& activeGlobals);

bool canMaterializeImmutableGlobalInitializer(
  go::GlobalOp globalOp,
  llvm::DenseMap<mlir::Value, bool>& memo,
  llvm::DenseSet<go::GlobalOp>& activeGlobals);

mlir::Value createGlobalString(mlir::Location loc, mlir::OpBuilder &builder,
                                     mlir::StringRef name, mlir::StringRef value,
                                     mlir::StringRef section,
                                     mlir::LLVM::Linkage linkage);

} // namespace mlir::go
