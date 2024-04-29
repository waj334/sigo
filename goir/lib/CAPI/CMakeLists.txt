add_mlir_public_c_api_library(CGoIR
        llvm/Passes.cxx
        mlir/Dialects.cxx
        mlir/Enums.cxx
        mlir/Operations.cxx
        mlir/Types.cxx
LINK_LIBS PUBLIC
    GoIR
)