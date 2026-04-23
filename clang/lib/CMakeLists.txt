add_library(GoClangSupport STATIC
        translation.cxx
)

llvm_update_compile_flags(GoClangSupport)

target_include_directories(GoClangSupport
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(GoClangSupport
        LINK_PUBLIC
        MLIRIR
        clangCIR
        clangFrontend
        clangAST
        clangBasic
        clangSerialization
        clangLex
)

install(TARGETS GoClangSupport ARCHIVE DESTINATION lib)

add_subdirectory(CAPI)
