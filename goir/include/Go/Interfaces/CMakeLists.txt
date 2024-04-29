set(LLVM_TARGET_DEFINITIONS ReflectTypeInterface.td)
mlir_tablegen(ReflectTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(ReflectTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRReflectTypeInterfaceIncGen)