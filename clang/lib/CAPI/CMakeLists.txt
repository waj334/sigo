add_library(CGoClang STATIC
        translation.cxx
        passes.cxx
        ops.cxx
)

llvm_update_compile_flags(CGoClang)

target_include_directories(CGoClang
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${MLIR_INCLUDE_DIRS}
        ${CLANG_INCLUDE_DIRS}
)

target_link_libraries(CGoClang
        PUBLIC
        GoClangSupport
        MLIRIR
        MLIRCAPIIR
        MLIRCIRTransforms
        clangCIRLoweringCommon
        clangCIRLoweringDirectToLLVM
        clangFrontend
        clangLex
        clangBasic
)

install(TARGETS CGoClang ARCHIVE DESTINATION lib)
