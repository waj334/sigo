#ifndef _MLIR_DIALECT_GO_TRANSFORMS_PASSES
#define _MLIR_DIALECT_GO_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AttachDebugInfoToAllocaPass : Pass<"attach-debug-info-to-alloca-pass", "::mlir::go::AllocaOp"> {
    let summary = "Generates debug information describing local variables.";
    let description = [{
        Generates debug information describing local variables.
    }];
    let dependentDialects = ["mlir::go::GoDialect", "mlir::LLVM::LLVMDialect"];
}

def AttachDebugInfoToConstantPass : Pass<"attach-debug-info-to-constant-pass", "::mlir::go::GlobalConstantOp"> {
    let summary = "Generates debug information describing constant values.";
    let description = [{
        Generates debug information describing constant values.
    }];
    let dependentDialects = ["mlir::go::GoDialect", "mlir::LLVM::LLVMDialect"];
}

def AttachDebugInfoToFuncPass : Pass<"attach-debug-info-to-func-pass", "::mlir::go::FuncOp"> {
    let summary = "Generates debug information describing functions.";
    let description = [{
        Generates debug information describing functions.
    }];
    let dependentDialects = ["mlir::go::GoDialect", "mlir::LLVM::LLVMDialect"];
}

def AttachDebugInfoToGlobalPass : Pass<"attach-debug-info-to-global-pass", "::mlir::go::GlobalOp"> {
    let summary = "Generates debug information describing globals.";
    let description = [{
        Generates debug information describing globals.
    }];
    let dependentDialects = ["mlir::go::GoDialect", "mlir::LLVM::LLVMDialect"];
}

def CallPass : Pass<"call-pass", "::mlir::ModuleOp"> {
    let summary = "Generates call wrapper functions for special calls.";
    let description = [{
        Generates call wrapper functions for special calls.
    }];
}

def FuncPass : Pass<"func-pass", "::mlir::go::FuncOp"> {
    let summary = "Function transforms.";
    let description = [{
        Inserts initializers for defer and trampolines for recover.
    }];
}

def GlobalConstantsPass : Pass<"global-constants-pass", "::mlir::ModuleOp"> {
    let summary = "Generates global constants for certain occurences of values.";
    let description = [{
        Creates global constants for repeated string value occurences. Deduplicates global constant
        values in the program.
    }];
    let dependentDialects = ["mlir::go::GoDialect", "mlir::LLVM::LLVMDialect"];
}

def GlobalInitializerPass : Pass<"global-initializer-pass", "::mlir::ModuleOp"> {
    let summary = "Global Initializer Pass - Lowers package initializers to constant globals";
    let description = [{
        Global Initializer Pass - Lowers package initializers to constant globals
    }];
}

def HeapEscapePass : Pass<"go-heap-escape-pass", "::mlir::go::FuncOp"> {
    let summary = "Analyzes stack allocations that escape the stack.";
    let description = [{
        Analyzes stack allocations that escape the stack.
    }];
}

def PreprocessingPass : Pass<"go-preprocessing-pass", "::mlir::ModuleOp"> {
    let summary = "Preprocesses the module.";
    let description = [{
        Preprocesses the module.
    }];
}

def ValueNormalizationFuncPass : Pass<"go-value-normalization-func-pass", "::mlir::go::FuncOp"> {
    let summary = "Normalizes values at the function level.";
    let description = [{
        Normalizes values at the function level.
    }];
    let dependentDialects = ["mlir::go::GoDialect"];
}

def ValueNormalizationGlobalPass : Pass<"go-value-normalization-global-pass", "::mlir::go::GlobalOp"> {
    let summary = "Normalizes values at the global level.";
    let description = [{
        Normalizes values at the global level.
    }];
    let dependentDialects = ["mlir::go::GoDialect"];
}

def LowerToCorePass : Pass<"lower-go-to-core", "::mlir::ModuleOp"> {
    let summary = "Lower Go IR to core dialects";
    let description = [{
        Lower Go IR to core dialects
    }];
}

def LowerToLLVMPass : Pass<"lower-go-to-llvm", "::mlir::ModuleOp"> {
    let summary = "Lower Go IR to LLVM IR";
    let description = [{
        Lower Go IR to LLVM IR
    }];
}

#endif // _MLIR_DIALECT_GO_TRANSFORMS_PASSES