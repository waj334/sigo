#include "go-clang/capi/translation.h"
#include "go-clang/translation.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/DLTI/DLTI.h>

// Storage for a heap-allocated CIRModuleResult.
// The opaque GoClangCIRModule.ptr points to one of these.
struct GoClangCIRModuleStorage {
  CIRModuleResult result;
};

GoClangCIRModule goClangLowerPreambleToMlir(const char *src, size_t srcLen,
                                             const char *triple,
                                             const char **includePaths,
                                             size_t numIncludePaths) {
  std::vector<std::string> paths;
  for (size_t i = 0; i < numIncludePaths; ++i) {
    paths.emplace_back(includePaths[i]);
  }
  auto *storage = new GoClangCIRModuleStorage{
      lowerPreambleToMlir(llvm::StringRef(src, srcLen), std::string(triple), paths)};
  if (!storage->result) {
    delete storage;
    return {nullptr};
  }
  return {storage};
}

bool goClangCIRModuleIsNull(GoClangCIRModule module) {
  return module.ptr == nullptr;
}

MlirContext goClangCIRModuleGetContext(GoClangCIRModule module) {
  auto *storage = static_cast<GoClangCIRModuleStorage *>(module.ptr);
  return wrap(storage->result.context.get());
}

MlirModule goClangCIRModuleGetModule(GoClangCIRModule module) {
  auto *storage = static_cast<GoClangCIRModuleStorage *>(module.ptr);
  return wrap(storage->result.module.get());
}

void goClangCIRModuleDestroy(GoClangCIRModule module) {
  delete static_cast<GoClangCIRModuleStorage *>(module.ptr);
}

void goClangRegisterDialects(MlirContext ctx) {
  auto *context = unwrap(ctx);
  context->getOrLoadDialect<mlir::DLTIDialect>();
  context->getOrLoadDialect<cir::CIRDialect>();
}
