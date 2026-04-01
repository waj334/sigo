package clang

/*
#include <go-clang/capi/translation.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

// CIRModule owns a CIR MLIRContext and the ModuleOp produced from a C preamble.
// Call Destroy when done to free the context and all CIR operations.
type CIRModule struct {
	raw C.GoClangCIRModule
}

// LowerPreambleToMlir compiles src (C source text) using the given target triple
// and system include paths, and returns a CIRModule owning the resulting CIR
// MLIR module. Returns nil on failure.
func LowerPreambleToMlir(src, triple string, includePaths []string) *CIRModule {
	cSrc := C.CString(src)
	defer C.free(unsafe.Pointer(cSrc))
	cTriple := C.CString(triple)
	defer C.free(unsafe.Pointer(cTriple))

	// Build C array of include path strings.
	var cPaths **C.char
	if len(includePaths) > 0 {
		pathPtrs := make([]*C.char, len(includePaths))
		for i, p := range includePaths {
			pathPtrs[i] = C.CString(p)
			defer C.free(unsafe.Pointer(pathPtrs[i]))
		}
		cPaths = &pathPtrs[0]
	}

	raw := C.goClangLowerPreambleToMlir(cSrc, C.size_t(len(src)), cTriple,
		cPaths, C.size_t(len(includePaths)))
	if bool(C.goClangCIRModuleIsNull(raw)) {
		return nil
	}
	return &CIRModule{raw: raw}
}

// IsNull returns true if the module handle is invalid.
func (m *CIRModule) IsNull() bool {
	return bool(C.goClangCIRModuleIsNull(m.raw))
}

// Context returns the MLIRContext that owns all CIR operations.
// The returned context is borrowed from the CIRModule — do not call Destroy on it.
func (m *CIRModule) Context() mlir.Context {
	raw := C.goClangCIRModuleGetContext(m.raw)
	return mlir.WrapExternalContext(unsafe.Pointer(raw.ptr))
}

// Module returns the MLIR ModuleOp containing the CIR operations.
// The returned module is borrowed from the CIRModule — do not call Destroy on it.
func (m *CIRModule) Module() mlir.Module {
	raw := C.goClangCIRModuleGetModule(m.raw)
	return mlir.WrapExternalModule(unsafe.Pointer(raw.ptr))
}

// MergeInto serializes the CIR module to MLIR text and re-parses it into ctx
// (which must have CIR dialects registered via RegisterDialects), then moves
// all top-level CIR operations into mod.
// Returns false if serialization or parsing fails.
func (m *CIRModule) MergeInto(ctx mlir.Context, mod mlir.Module) bool {
	// Serialize the CIR module to MLIR assembly text with location info
	// so that source locations (from #line directives) survive re-parsing.
	flags := mlir.NewOpPrintingFlags().WithEnableDebugInfo(true, false)
	defer flags.Destroy()
	cirText := m.Module().Operation().StringWithFlags(flags)

	// Parse the text back into the target context, which must have the CIR
	// dialect registered so the operation names and types are known.
	parsed := mlir.NewModuleFromString(ctx, cirText)
	if parsed.IsNull() {
		return false
	}
	defer parsed.Destroy()

	srcBlock := parsed.Body()
	dstBlock := mod.Body()
	for srcOp := srcBlock.FirstOperation(); !srcOp.IsNull(); srcOp = srcOp.NextInBlock() {
		dstBlock.AppendOwnedOperation(srcOp.Clone())
	}
	return true
}

// RegisterDialects loads the CIR dialect and its dependencies (DLTI) into ctx
// so that CIR operations can be created and parsed in that context.
// Call this before mlir.Context.LoadAllAvailableDialects().
func RegisterDialects(ctx mlir.Context) {
	C.goClangRegisterDialects(C.MlirContext{ptr: ctx.Ptr()})
}

// Destroy frees the owned context and all CIR operations.
// The CIRModule must not be used after this call.
func (m *CIRModule) Destroy() {
	C.goClangCIRModuleDestroy(m.raw)
	m.raw = C.GoClangCIRModule{}
}
