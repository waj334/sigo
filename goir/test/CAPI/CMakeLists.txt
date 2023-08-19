add_mlir_aggregate(CGoIRTestLib
        STATIC
        EMBED_LIBS
        MLIRCAPIIR
        # TODO: Remove this in favor of showing fine grained dialect registration
        # (once available).
        MLIRCAPIRegisterEverything
        CGoIR
)

add_llvm_executable(CGoIR-test
        cgoir-test.c
)
llvm_update_compile_flags(CGoIR-test)
target_link_libraries(CGoIR-test
        PRIVATE CGoIRTestLib)
