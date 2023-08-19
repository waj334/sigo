get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        GoIR
)
add_llvm_executable(go-opt go-opt.cxx)

llvm_update_compile_flags(go-opt)
target_link_libraries(go-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(go-opt)
