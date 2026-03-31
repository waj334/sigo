#ifndef GO_CLANG_CAPI_TRANSLATION_H
#define GO_CLANG_CAPI_TRANSLATION_H

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque handle owning a CIR MLIRContext and ModuleOp.
  // Must be destroyed with goClangCIRModuleDestroy when no longer needed.
  typedef struct GoClangCIRModule
  {
    void *ptr;
  } GoClangCIRModule;

  // Compile srcLen bytes of C source text (src) using the given target triple
  // and return a handle owning the resulting CIR module.
  // Returns a null handle on failure.
  GoClangCIRModule goClangLowerPreambleToMlir(const char *src, size_t srcLen,
                                               const char *triple);

  // Returns true when the handle is null (compilation failed).
  bool goClangCIRModuleIsNull(GoClangCIRModule module);

  // Return the MLIRContext owned by the module.
  // The returned context is borrowed — do NOT call mlirContextDestroy on it.
  MlirContext goClangCIRModuleGetContext(GoClangCIRModule module);

  // Return the MLIR ModuleOp containing the CIR operations.
  // The returned module is borrowed — do NOT call mlirModuleDestroy on it.
  MlirModule goClangCIRModuleGetModule(GoClangCIRModule module);

  // Destroy the handle and free the owned context and all CIR operations.
  void goClangCIRModuleDestroy(GoClangCIRModule module);

  // Register CIR and required dialects (DLTIDialect, etc.) into ctx so that
  // CIR MLIR text can be parsed and CIR operations are usable in that context.
  void goClangRegisterDialects(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // GO_CLANG_CAPI_TRANSLATION_H
