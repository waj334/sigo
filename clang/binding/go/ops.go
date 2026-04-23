package clang

/*
#include <go-clang/capi/ops.h>
#include <stdlib.h>
*/
import "C"
import (
	"strings"
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

//===----------------------------------------------------------------------===//
// CIR type constructors
//===----------------------------------------------------------------------===//

// NewCIRIntType returns the CIR integer type !sNi (signed) or !uNi (unsigned).
func NewCIRIntType(ctx mlir.Context, width uint, signed bool) mlir.Type {
	return wrapType(C.goClangCreateCIRIntType(unwrapContext(ctx), C.uint(width), C.bool(signed)))
}

// NewCIRVoidType returns the CIR void type !cir.void.
func NewCIRVoidType(ctx mlir.Context) mlir.Type {
	return wrapType(C.goClangCreateCIRVoidType(unwrapContext(ctx)))
}

// CIRTypeIsVoid returns true if t is a CIR void type.
func CIRTypeIsVoid(t mlir.TypeLike) bool {
	return bool(C.goClangCIRTypeIsVoid(unwrapType(t)))
}

// NewCIRPtrType returns the CIR pointer type !cir.ptr<pointee>.
func NewCIRPtrType(ctx mlir.Context, pointee mlir.Type) mlir.Type {
	return wrapType(C.goClangCreateCIRPtrType(unwrapContext(ctx), unwrapType(pointee)))
}

// NewCIRBoolType returns the CIR bool type !cir.bool.
func NewCIRBoolType(ctx mlir.Context) mlir.Type {
	return wrapType(C.goClangCreateCIRBoolType(unwrapContext(ctx)))
}

// NewCIRFloatType returns the CIR single-precision float type !cir.float.
func NewCIRFloatType(ctx mlir.Context) mlir.Type {
	return wrapType(C.goClangCreateCIRFloatType(unwrapContext(ctx)))
}

// NewCIRDoubleType returns the CIR double-precision float type !cir.double.
func NewCIRDoubleType(ctx mlir.Context) mlir.Type {
	return wrapType(C.goClangCreateCIRDoubleType(unwrapContext(ctx)))
}

//===----------------------------------------------------------------------===//
// CIR operation constructors
//===----------------------------------------------------------------------===//

// NewCIRCallOp creates a cir.call operation:
//
//	cir.call @callee(operands...) : (paramTypes...) -> resultType
//
// Pass NewCIRVoidType for void-returning functions; no result is added in that
// case.
func NewCIRCallOp(loc mlir.LocationLike, callee string, resultType mlir.TypeLike, operands []mlir.ValueLike) mlir.Operation {
	cCallee := C.CString(callee)
	defer C.free(unsafe.Pointer(cCallee))

	var opPtr *C.MlirValue
	if len(operands) > 0 {
		raw := make([]C.MlirValue, len(operands))
		for i, v := range operands {
			raw[i] = unwrapValue(v)
		}
		opPtr = &raw[0]
	}

	return wrapOperation(C.goClangCreateCIRCallOp(
		unwrapLocation(loc),
		cCallee, C.size_t(len(callee)),
		unwrapType(resultType),
		C.intptr_t(len(operands)),
		opPtr,
	))
}

//===----------------------------------------------------------------------===//
// CIR function-signature extraction
//===----------------------------------------------------------------------===//

// CIRFuncSig holds the extracted signature of one public cir.func operation.
type CIRFuncSig struct {
	Name       string
	ParamTypes []mlir.Type
	ReturnType mlir.Type // void type when the function returns void
}

// ExtractCIRFuncSigs walks mod and returns signatures of all non-private
// cir.func operations. Call this BEFORE MergeInto while the CIRModule is still
// valid.
func ExtractCIRFuncSigs(mod mlir.Module) []CIRFuncSig {
	raw := C.goClangExtractCIRFuncSigs(unwrapModule(mod))
	defer C.goClangCIRFuncSigsDestroy(raw)

	n := int(C.goClangCIRFuncSigsCount(raw))
	sigs := make([]CIRFuncSig, n)
	for i := 0; i < n; i++ {
		idx := C.intptr_t(i)

		// Name.
		var nameLen C.size_t
		namePtr := C.goClangCIRFuncSigName(raw, idx, &nameLen)
		sigs[i].Name = C.GoStringN(namePtr, C.int(nameLen))

		// Parameter types.
		nParams := int(C.goClangCIRFuncSigParamCount(raw, idx))
		sigs[i].ParamTypes = make([]mlir.Type, nParams)
		for p := 0; p < nParams; p++ {
			sigs[i].ParamTypes[p] = wrapType(C.goClangCIRFuncSigParamType(raw, idx, C.intptr_t(p)))
		}

		// Return type.
		sigs[i].ReturnType = wrapType(C.goClangCIRFuncSigReturnType(raw, idx))
	}
	return sigs
}

//===----------------------------------------------------------------------===//
// Preprocessor macro extraction
//===----------------------------------------------------------------------===//

// MacroDef holds a single numeric macro definition extracted from a C preamble.
type MacroDef struct {
	Name  string
	Value string
}

// DumpMacros runs the Clang preprocessor on src (C source) for the given
// target triple and returns all macros that expand to integer or float
// literals. Returns nil on failure.
func DumpMacros(src, triple string) []MacroDef {
	cSrc := C.CString(src)
	defer C.free(unsafe.Pointer(cSrc))
	cTriple := C.CString(triple)
	defer C.free(unsafe.Pointer(cTriple))

	dump := C.goClangDumpMacros(cSrc, C.size_t(len(src)), cTriple)
	if dump == nil {
		return nil
	}
	defer C.goClangFreeMacroDump(dump)

	text := C.GoString(dump)
	if text == "" {
		return nil
	}

	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	defs := make([]MacroDef, 0, len(lines))
	for _, line := range lines {
		if line == "" {
			continue
		}
		tab := strings.IndexByte(line, '\t')
		if tab < 0 {
			continue
		}
		defs = append(defs, MacroDef{Name: line[:tab], Value: line[tab+1:]})
	}
	return defs
}
