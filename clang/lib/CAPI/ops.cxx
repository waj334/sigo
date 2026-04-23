#include "go-clang/capi/ops.h"

#include <clang/CIR/Dialect/Builder/CIRBaseBuilder.h>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Frontend/FrontendActions.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// CIR type constructors
//===----------------------------------------------------------------------===//

MlirType goClangCreateCIRIntType(MlirContext ctx, unsigned width, bool isSigned) {
  return wrap(cir::IntType::get(unwrap(ctx), width, isSigned));
}

MlirType goClangCreateCIRVoidType(MlirContext ctx) {
  return wrap(cir::VoidType::get(unwrap(ctx)));
}

bool goClangCIRTypeIsVoid(MlirType t) {
  return mlir::isa<cir::VoidType>(unwrap(t));
}

MlirType goClangCreateCIRPtrType(MlirContext ctx, MlirType pointee) {
  return wrap(cir::PointerType::get(unwrap(ctx), unwrap(pointee)));
}

MlirType goClangCreateCIRBoolType(MlirContext ctx) {
  return wrap(cir::BoolType::get(unwrap(ctx)));
}

MlirType goClangCreateCIRFloatType(MlirContext ctx) {
  return wrap(cir::SingleType::get(unwrap(ctx)));
}

MlirType goClangCreateCIRDoubleType(MlirContext ctx) {
  return wrap(cir::DoubleType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// CIR operation constructors
//===----------------------------------------------------------------------===//

MlirOperation goClangCreateCIRCallOp(MlirLocation loc,
                                      const char *callee, size_t calleeLen,
                                      MlirType resultType,
                                      intptr_t nOperands,
                                      MlirValue const *operands) {
  auto *ctx = unwrap(loc).getContext();

  // Build the operation state.
  mlir::OperationState state(unwrap(loc), "cir.call");

  // Callee attribute.
  auto sym = mlir::FlatSymbolRefAttr::get(ctx,
                                          llvm::StringRef(callee, calleeLen));
  state.addAttribute("callee", sym);

  // Operands.
  llvm::SmallVector<mlir::Value> ops;
  ops.reserve(nOperands);
  for (intptr_t i = 0; i < nOperands; ++i)
    ops.push_back(unwrap(operands[i]));
  state.addOperands(ops);

  // Result type — only added for non-void return types.
  auto resTy = unwrap(resultType);
  if (resTy && !mlir::isa<cir::VoidType>(resTy))
    state.addTypes(resTy);

  return wrap(mlir::Operation::create(state));
}

//===----------------------------------------------------------------------===//
// CIR function-signature extraction
//===----------------------------------------------------------------------===//

struct GoClangCIRFuncSigsStorage {
  struct Entry {
    std::string name;
    std::vector<mlir::Type> params;
    mlir::Type returnType;
  };
  std::vector<Entry> entries;
};

GoClangCIRFuncSigs goClangExtractCIRFuncSigs(MlirModule mod) {
  auto *storage = new GoClangCIRFuncSigsStorage();
  auto moduleOp = unwrap(mod);

  // Walk all cir.func ops. Include both definitions and declarations
  // (from #include'd headers) so CGo stubs can be generated for external
  // C functions.
  moduleOp.walk([&](cir::FuncOp fn) {

    GoClangCIRFuncSigsStorage::Entry entry;
    entry.name = fn.getSymName().str();

    cir::FuncType fnTy = fn.getFunctionType();
    for (mlir::Type t : fnTy.getInputs())
      entry.params.push_back(t);

    // getReturnType() returns null for void functions; normalise to VoidType.
    mlir::Type ret = fnTy.getReturnType();
    if (!ret)
      ret = cir::VoidType::get(fn.getContext());
    entry.returnType = ret;

    storage->entries.push_back(std::move(entry));
  });

  return {storage};
}

intptr_t goClangCIRFuncSigsCount(GoClangCIRFuncSigs sigs) {
  return static_cast<intptr_t>(
      static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr)->entries.size());
}

const char *goClangCIRFuncSigName(GoClangCIRFuncSigs sigs, intptr_t i,
                                   size_t *nameLen) {
  auto &entry =
      static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr)->entries[i];
  *nameLen = entry.name.size();
  return entry.name.c_str();
}

intptr_t goClangCIRFuncSigParamCount(GoClangCIRFuncSigs sigs, intptr_t i) {
  auto &entry =
      static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr)->entries[i];
  return static_cast<intptr_t>(entry.params.size());
}

MlirType goClangCIRFuncSigParamType(GoClangCIRFuncSigs sigs, intptr_t i,
                                     intptr_t p) {
  auto &entry =
      static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr)->entries[i];
  return wrap(entry.params[p]);
}

MlirType goClangCIRFuncSigReturnType(GoClangCIRFuncSigs sigs, intptr_t i) {
  const auto &entry =
      static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr)->entries[i];
  return wrap(entry.returnType);
}

void goClangCIRFuncSigsDestroy(GoClangCIRFuncSigs sigs) {
  delete static_cast<GoClangCIRFuncSigsStorage *>(sigs.ptr);
}

//===----------------------------------------------------------------------===//
// Preprocessor macro extraction
//===----------------------------------------------------------------------===//

char *goClangDumpMacros(const char *src, size_t srcLen, const char *triple) {
  // We run Clang in preprocessor-only mode with -dM to collect all macro
  // definitions, then filter to those that expand to integer or float literals.
  auto ci = std::make_unique<clang::CompilerInstance>();
  ci->createDiagnostics();
  ci->getTargetOpts().Triple = triple;

  std::vector<std::string> includes;
  clang::LangOptions::setLangDefaults(ci->getLangOpts(), clang::Language::C,
                                      llvm::Triple(triple), includes,
                                      clang::LangStandard::lang_c17);

  // Ask the preprocessor to dump macro definitions.
  ci->getPreprocessorOutputOpts().ShowMacros = 1;
  ci->getPreprocessorOutputOpts().ShowIncludeDirectives = 0;

  auto memBuf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(src, srcLen), "preamble.c");
  ci->getPreprocessorOpts().addRemappedFile("preamble.c", memBuf.release());
  ci->getFrontendOpts().Inputs.emplace_back(
      "preamble.c", clang::InputKind(clang::Language::C));
  ci->getFrontendOpts().ProgramAction = clang::frontend::PrintPreprocessedInput;

  // Redirect preprocessor output to a string.
  std::string output;
  llvm::raw_string_ostream os(output);
  ci->createFileManager();
  ci->createSourceManager();

  // Use PrintPreprocessedAction to get macro dump output.
  clang::PrintPreprocessedAction action;
  // Redirect standard output to our string stream.
  ci->getFrontendOpts().OutputFile = "-"; // stdout
  // We'll capture via the stream redirect below. For simplicity, run the
  // action and parse its stdout. Since we can't easily redirect CI's stdout,
  // use a different approach: manually invoke the preprocessor.
  // Actually, use the token output approach via a pipe is complex.
  // Instead, enumerate macros via the Preprocessor API after parsing.
  ci->createPreprocessor(clang::TU_Complete);
  auto &pp = ci->getPreprocessor();

  // Run the preprocessor over the input to populate macro tables.
  ci->getDiagnosticClient().BeginSourceFile(ci->getLangOpts(), &pp);
  pp.EnterMainSourceFile();
  // Lex all tokens to force macro expansion and registration.
  clang::Token tok;
  do {
    pp.Lex(tok);
  } while (tok.isNot(clang::tok::eof));
  ci->getDiagnosticClient().EndSourceFile();

  // Build output string: one "NAME\tvalue\n" per integer/float macro.
  std::string result;
  // Walk the macro table.
  for (auto it = pp.macro_begin(); it != pp.macro_end(); ++it) {
    const clang::IdentifierInfo *id = it->first;
    auto *md = it->second.getLatest();
    if (!md || !md->isDefined())
      continue;
    const clang::MacroInfo *mi = md->getMacroInfo();
    if (!mi || mi->isFunctionLike() || mi->getNumTokens() != 1)
      continue;

    const clang::Token &valTok = mi->getReplacementToken(0);
    if (valTok.is(clang::tok::numeric_constant)) {
      llvm::StringRef spelling = pp.getSpelling(valTok);
      // Accept only tokens that look like integer or float literals (no
      // trailing 'u', 'l', etc. after digits — keep it simple).
      bool valid = !spelling.empty();
      for (char c : spelling) {
        if (!std::isdigit(c) && c != '.' && c != 'e' && c != 'E' &&
            c != '+' && c != '-' && c != 'x' && c != 'X' &&
            c != 'a' && c != 'A' && c != 'b' && c != 'B' &&
            c != 'c' && c != 'C' && c != 'd' && c != 'D' &&
            c != 'f' && c != 'F') {
          // Suffix character like U, L, LL — skip this macro.
          if (std::toupper(c) == 'U' || std::toupper(c) == 'L' ||
              std::toupper(c) == 'N') {
            // Integer suffix — strip suffix and still keep the numeric part.
            // We'll let Go parse it as an untyped constant.
            break;
          }
          valid = false;
          break;
        }
      }
      if (!valid)
        continue;
      result += id->getName().str();
      result += '\t';
      result += spelling.str();
      result += '\n';
    }
  }

  // Return heap-allocated copy.
  char *buf = static_cast<char *>(std::malloc(result.size() + 1));
  if (!buf)
    return nullptr;
  std::memcpy(buf, result.c_str(), result.size() + 1);
  return buf;
}

void goClangFreeMacroDump(char *dump) { std::free(dump); }

