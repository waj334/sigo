#include "go-clang/translation.h"

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/LangOptions.h>
#include <clang/CIR/CIRGenerator.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Support/MemoryBuffer.h>


namespace {

// Subclass CIRGenerator to expose the protected mlirContext member so we can
// transfer its ownership before the generator is destroyed by Clang's pipeline.
class ExtractableCIRGenerator : public cir::CIRGenerator {
public:
  using cir::CIRGenerator::CIRGenerator;

  std::unique_ptr<mlir::MLIRContext> extractContext() {
    return std::move(this->mlirContext);
  }
};

// Custom ASTFrontendAction that captures the CIR module in EndSourceFileAction,
// which fires after HandleTranslationUnit (module fully built) but before
// CompilerInstance destroys the ASTConsumer.
class CIRCaptureAction : public clang::ASTFrontendAction {
  ExtractableCIRGenerator *rawGen = nullptr;
  CIRModuleResult result;

public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef /*InFile*/) override {
    auto gen = std::make_unique<ExtractableCIRGenerator>(
        CI.getDiagnostics(), &CI.getVirtualFileSystem(),
        CI.getCodeGenOpts());
    rawGen = gen.get();
    return gen;
  }

  void EndSourceFileAction() override {
    if (!rawGen)
      return;
    mlir::ModuleOp mod = rawGen->getModule();
    if (!mod)
      return;
    // Move the MLIRContext out before CIRGenerator is destroyed.
    // Members in CIRGenerator are destroyed in reverse declaration order:
    // cgm first, then mlirContext — so cgm's destructor still runs against a
    // live context (owned by us), keeping destruction safe.
    result.context = rawGen->extractContext();
    result.module = mlir::OwningOpRef<mlir::ModuleOp>(mod);
  }

  CIRModuleResult takeResult() { return std::move(result); }
};

} // namespace

CIRModuleResult lowerPreambleToMlir(const llvm::StringRef preambleSrc,
                                    const std::string &triple) {
  auto ci = std::make_unique<clang::CompilerInstance>();
  ci->createDiagnostics();

  ci->getTargetOpts().Triple = triple;

  // Feed preamble source from memory rather than disk.
  auto memBuf =
      llvm::MemoryBuffer::getMemBufferCopy(preambleSrc, "preamble.c");
  ci->getPreprocessorOpts().addRemappedFile("preamble.c", memBuf.release());
  ci->getFrontendOpts().Inputs.emplace_back(
      "preamble.c", clang::InputKind(clang::Language::C));

  CIRCaptureAction action;
  ci->ExecuteAction(action);
  return action.takeResult();
}
