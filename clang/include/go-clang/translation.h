#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <memory>
#include <string>

// Holds the MLIR context and CIR module produced from a C preamble.
// The module is only valid while the context is alive — keep them together.
struct CIRModuleResult {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  explicit operator bool() const { return context && module; }
};

// Compile preambleSrc (C source text) into a CIR MLIR module.
// includePaths are added as system include directories.
// Returns an empty CIRModuleResult on failure.
CIRModuleResult lowerPreambleToMlir(llvm::StringRef preambleSrc,
                                    const std::string &triple,
                                    const std::vector<std::string> &includePaths = {});
