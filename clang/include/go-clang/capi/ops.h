#ifndef GO_CLANG_CAPI_OPS_H
#define GO_CLANG_CAPI_OPS_H

#include <mlir-c/IR.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// CIR type constructors
//===----------------------------------------------------------------------===//

// Returns the CIR integer type !sNi (signed) or !uNi (unsigned).
MlirType goClangCreateCIRIntType(MlirContext ctx, unsigned width, bool isSigned);

// Returns the CIR void type !cir.void.
MlirType goClangCreateCIRVoidType(MlirContext ctx);

// Returns true if t is a CIR void type.
bool goClangCIRTypeIsVoid(MlirType t);

// Returns the CIR pointer type !cir.ptr<pointee>.
MlirType goClangCreateCIRPtrType(MlirContext ctx, MlirType pointee);

// Returns the CIR bool type !cir.bool.
MlirType goClangCreateCIRBoolType(MlirContext ctx);

// Returns the CIR single-precision float type !cir.float.
MlirType goClangCreateCIRFloatType(MlirContext ctx);

// Returns the CIR double-precision float type !cir.double.
MlirType goClangCreateCIRDoubleType(MlirContext ctx);

//===----------------------------------------------------------------------===//
// CIR operation constructors
//===----------------------------------------------------------------------===//

// Creates a cir.call operation.
//   cir.call @callee(operands...) : (paramTypes...) -> resultType
// Pass goClangCreateCIRVoidType for void-returning functions; no result is
// added in that case.
MlirOperation goClangCreateCIRCallOp(MlirLocation loc,
                                      const char *callee, size_t calleeLen,
                                      MlirType resultType,
                                      intptr_t nOperands,
                                      MlirValue const *operands);

//===----------------------------------------------------------------------===//
// CIR function-signature extraction
//
// Walk a CIR ModuleOp and collect signatures of all non-private cir.func ops.
// Extract BEFORE calling MergeInto() while the CIRModule is still valid.
//===----------------------------------------------------------------------===//

// Opaque handle owning an array of extracted CIR function signatures.
typedef struct GoClangCIRFuncSigs {
  void *ptr;
} GoClangCIRFuncSigs;

// Walk mod and return the signatures of all non-private cir.func operations.
GoClangCIRFuncSigs goClangExtractCIRFuncSigs(MlirModule mod);

// Returns the number of function signatures in sigs.
intptr_t goClangCIRFuncSigsCount(GoClangCIRFuncSigs sigs);

// Returns the name of function i. *nameLen is set to the byte length.
const char *goClangCIRFuncSigName(GoClangCIRFuncSigs sigs, intptr_t i,
                                   size_t *nameLen);

// Returns the number of parameters of function i.
intptr_t goClangCIRFuncSigParamCount(GoClangCIRFuncSigs sigs, intptr_t i);

// Returns the CIR type of parameter p of function i.
MlirType goClangCIRFuncSigParamType(GoClangCIRFuncSigs sigs, intptr_t i,
                                     intptr_t p);

// Returns the CIR return type of function i.
// This may be the void type (check with goClangCIRTypeIsVoid).
MlirType goClangCIRFuncSigReturnType(GoClangCIRFuncSigs sigs, intptr_t i);

// Destroys the handle returned by goClangExtractCIRFuncSigs.
void goClangCIRFuncSigsDestroy(GoClangCIRFuncSigs sigs);

//===----------------------------------------------------------------------===//
// Preprocessor macro extraction
//
// Run the Clang preprocessor on a C preamble and return the set of macros
// that expand to integer or floating-point literals.
//===----------------------------------------------------------------------===//

// Run the Clang preprocessor on srcLen bytes of C source (triple = target
// triple) and return a heap-allocated, NUL-terminated string containing one
// "NAME\tvalue\n" line per macro that expands to a simple integer or float
// literal.  Returns NULL on failure.  Free the result with
// goClangFreeMacroDump.
char *goClangDumpMacros(const char *src, size_t srcLen, const char *triple);

// Free a string returned by goClangDumpMacros.
void goClangFreeMacroDump(char *dump);

#ifdef __cplusplus
}
#endif

#endif // GO_CLANG_CAPI_OPS_H
