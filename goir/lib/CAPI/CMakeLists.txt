add_mlir_public_c_api_library(CGoIR
        mlir/Dialects.cxx
        mlir/Enums.cxx
        mlir/Operations.cxx
        mlir/Passes.cxx
        mlir/Types.cxx
        LINK_LIBS PUBLIC
        GoIR
)