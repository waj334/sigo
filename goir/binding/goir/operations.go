package goir

/*
#include <Go-c/mlir/Operations.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"

	"pkg.si-go.dev/go-mlir/mlir"
)

//===----------------------------------------------------------------------===//
// ASM Operations
//===----------------------------------------------------------------------===//

func NewInlineAssemblyOperation(ctx mlir.Context, asmStrAttr mlir.AttributeLike,
	constraints []mlir.AttributeLike,
	registerClobbers []mlir.AttributeLike,
	operands []mlir.ValueLike,
	location mlir.LocationLike,
) mlir.Operation {
	return wrapOperation(C.mlirGoCreateInlineAssemblyOperation(
		unwrapContext(ctx),
		unwrapAttribute(asmStrAttr),
		C.int(len(constraints)),
		unwrapAttributeSlice(constraints),
		C.int(len(registerClobbers)),
		unwrapAttributeSlice(registerClobbers),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

func NewAddCOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAddCOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewAddFOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAddFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewAddIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAddIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewAddStrOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAddStrOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewAndOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAndOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewAndNotOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAndNotOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpCOperation(ctx mlir.Context, resultType mlir.TypeLike, predicate mlir.AttributeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpCOperation(unwrapContext(ctx), unwrapType(resultType), unwrapAttribute(predicate), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpFOperation(ctx mlir.Context, resultType mlir.TypeLike, predicate mlir.AttributeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapAttribute(predicate), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpIOperation(ctx mlir.Context, resultType mlir.TypeLike, predicate mlir.AttributeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapAttribute(predicate), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpInterfaceOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpInterfaceOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpStringOperation(ctx mlir.Context, resultType mlir.TypeLike, predicate mlir.AttributeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpStringOperation(unwrapContext(ctx), unwrapType(resultType), unwrapAttribute(predicate), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewCmpNilOperation(ctx mlir.Context, resultType mlir.TypeLike, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCmpNilOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapLocation(location)))
}

func NewDivCOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDivCOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewDivFOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDivFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewDivSIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDivSIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewDivUIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDivUIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewMulCOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMulCOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewMulFOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMulFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewMulIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMulIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewOrOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateOrOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewRemFOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateRemFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewRemSIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateRemSIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewRemUIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateRemUIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewShlOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateShlOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewShrUIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateShrUIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewShrSIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateShrSIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewSubCOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSubCOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewSubFOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSubFOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewSubIOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSubIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

func NewXorOperation(ctx mlir.Context, resultType mlir.TypeLike, x, y mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateXorOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(x), unwrapValue(y), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

func NewComplementOperation(ctx mlir.Context, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateComplementOperation(unwrapContext(ctx), unwrapValue(x), unwrapLocation(location)))
}

func NewNegCOperation(ctx mlir.Context, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNegCOperation(unwrapContext(ctx), unwrapValue(x), unwrapLocation(location)))
}

func NewNegFOperation(ctx mlir.Context, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNegFOperation(unwrapContext(ctx), unwrapValue(x), unwrapLocation(location)))
}

func NewNegIOperation(ctx mlir.Context, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNegIOperation(unwrapContext(ctx), unwrapValue(x), unwrapLocation(location)))
}

func NewNotOperation(ctx mlir.Context, x mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNotOperation(unwrapContext(ctx), unwrapValue(x), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Map Operations
//===----------------------------------------------------------------------===//

func NewMapAddrOperation(ctx mlir.Context, resultType mlir.TypeLike, mapVal, key mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMapAddrOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(mapVal), unwrapValue(key), unwrapLocation(location)))
}

func NewMapLookupOperation(ctx mlir.Context, resultType mlir.TypeLike, mapVal, key mlir.ValueLike, hasOk bool, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMapLookupOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(mapVal), unwrapValue(key), C.bool(hasOk), unwrapLocation(location)))
}

func NewMapRangeOp(ctx mlir.Context, value mlir.ValueLike, bodyDest, exitDest mlir.Block, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMapRangeOp(unwrapContext(ctx), unwrapValue(value), unwrapBlock(bodyDest), unwrapBlock(exitDest), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

func NewAllocaOperation(ctx mlir.Context, resultType, elementType mlir.TypeLike, numElements int, isHeap bool, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAllocaOperation(unwrapContext(ctx), unwrapType(resultType), unwrapType(elementType), C.int(numElements), C.bool(isHeap), unwrapLocation(location)))
}

func AllocaOperationSetName(op mlir.Operation, name string) {
	C.mlirGoAllocaOperationSetName(unwrapOperation(op), unwrapStringRef(mlir.NewStringRef(name)))
}

func AllocaOperationSetIsHeap(op mlir.Operation, isHeap bool) {
	C.mlirGoAllocaOperationSetIsHeap(unwrapOperation(op), C.bool(isHeap))
}

func NewLoadOperation(ctx mlir.Context, x mlir.ValueLike, resultType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateLoadOperation(unwrapContext(ctx), unwrapValue(x), unwrapType(resultType), unwrapLocation(location)))
}

func NewVolatileLoadOperation(ctx mlir.Context, x mlir.ValueLike, resultType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateVolatileLoadOperation(unwrapContext(ctx), unwrapValue(x), unwrapType(resultType), unwrapLocation(location)))
}

func NewAtomicLoadOperation(ctx mlir.Context, x mlir.ValueLike, resultType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAtomicLoadOperation(unwrapContext(ctx), unwrapValue(x), unwrapType(resultType), unwrapLocation(location)))
}

func NewStoreOperation(ctx mlir.Context, value, address mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateStoreOperation(unwrapContext(ctx), unwrapValue(value), unwrapValue(address), unwrapLocation(location)))
}

func NewVolatileStoreOperation(ctx mlir.Context, value, address mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateVolatileStoreOperation(unwrapContext(ctx), unwrapValue(value), unwrapValue(address), unwrapLocation(location)))
}

func NewAtomicStoreOperation(ctx mlir.Context, value, address mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAtomicStoreOperation(unwrapContext(ctx), unwrapValue(value), unwrapValue(address), unwrapLocation(location)))
}

func NewGepOperation(ctx mlir.Context, addr mlir.ValueLike, baseType mlir.TypeLike, constIndices []int, dynamicIndices []mlir.ValueLike, indexFlags []bool, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	var ptrConstIndices *C.int32_t
	if len(constIndices) > 0 {
		ss := make([]C.int32_t, len(constIndices))
		for i, v := range constIndices {
			ss[i] = C.int32_t(v)
		}
		ptrConstIndices = (*C.int32_t)(unsafe.Pointer(unsafe.SliceData(ss)))
	}

	var ptrIndexFlags *C.bool
	if len(indexFlags) > 0 {
		ss := make([]C.bool, len(indexFlags))
		for i, v := range indexFlags {
			ss[i] = C.bool(v)
		}
		ptrIndexFlags = (*C.bool)(unsafe.Pointer(unsafe.SliceData(ss)))
	}

	return wrapOperation(C.mlirGoCreateGepOperation(
		unwrapContext(ctx),
		unwrapValue(addr),
		unwrapType(baseType),
		C.int(len(constIndices)),
		ptrConstIndices,
		C.int(len(dynamicIndices)),
		unwrapValueSlice(dynamicIndices),
		C.int(len(indexFlags)),
		ptrIndexFlags,
		unwrapType(typ),
		unwrapLocation(location),
	))
}

func NewGlobalOperation(ctx mlir.Context, linkage mlir.AttributeLike, symbol string, section string, alignment int64, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	var alignmentAttr mlir.AttributeLike
	if alignment != 0 {
		alignmentAttr = mlir.NewIntegerAttr(mlir.NewIntegerType(ctx, 64), alignment)
	}

	return wrapOperation(C.mlirGoCreateGlobalOperation(
		unwrapContext(ctx),
		unwrapAttribute(linkage),
		unwrapStringRef(mlir.NewStringRef(symbol)),
		unwrapStringRef(mlir.NewStringRef(section)),
		unwrapAttribute(alignmentAttr),
		unwrapType(typ),
		unwrapLocation(location),
	))
}

func NewYieldOperation(ctx mlir.Context, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateYieldOperation(unwrapContext(ctx), unwrapValue(value), unwrapLocation(location)))
}

func NewSliceOperation(ctx mlir.Context, input mlir.ValueLike, low, high, max mlir.ValueLike, resultType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSliceOperation(
		unwrapContext(ctx),
		unwrapValue(input),
		unwrapValue(low),
		unwrapValue(high),
		unwrapValue(max),
		unwrapType(resultType),
		unwrapLocation(location),
	))
}

func NewAddressOfOperation(ctx mlir.Context, symbol string, resultType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAddressOfOperation(unwrapContext(ctx), unwrapStringRef(mlir.NewStringRef(symbol)), unwrapType(resultType), unwrapLocation(location)))
}

func NewNilPointerCheckOperation(ctx mlir.Context, addr mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNilPointerCheckOperation(unwrapContext(ctx), unwrapValue(addr), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Slice Operations
//===----------------------------------------------------------------------===//

func NewSliceAddrOperation(ctx mlir.Context, resultType mlir.TypeLike, slice, index mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSliceAddrOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(slice), unwrapValue(index), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

func NewStringAddrOperation(ctx mlir.Context, resultType mlir.TypeLike, slice, index mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateStringAddrOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(slice), unwrapValue(index), unwrapLocation(location)))
}

func NewStringRangeOp(ctx mlir.Context, value mlir.ValueLike, bodyDest, exitDest mlir.Block, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateStringRangeOp(unwrapContext(ctx), unwrapValue(value), unwrapBlock(bodyDest), unwrapBlock(exitDest), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Struct Operations
//===----------------------------------------------------------------------===//

func NewExtractOperation(ctx mlir.Context, index uint64, fieldType mlir.TypeLike, structValue mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateExtractOperation(unwrapContext(ctx), C.uint64_t(index), unwrapType(fieldType), unwrapValue(structValue), unwrapLocation(location)))
}

func NewInsertOperation(ctx mlir.Context, index uint64, value, structValue mlir.ValueLike, structType mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateInsertOperation(unwrapContext(ctx), C.uint64_t(index), unwrapValue(value), unwrapValue(structValue), unwrapType(structType), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

func NewConstantOperation(ctx mlir.Context, value, symbol mlir.AttributeLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateConstantOperation(unwrapContext(ctx), unwrapAttribute(value), unwrapAttribute(symbol), unwrapType(typ), unwrapLocation(location)))
}

func NewGlobalConstantOperation(ctx mlir.Context, value, symbol mlir.AttributeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGlobalConstantOperation(unwrapContext(ctx), unwrapAttribute(value), unwrapAttribute(symbol), unwrapLocation(location)))
}

func GlobalConstantOperationAddBody(op mlir.Operation, body mlir.Block) {
	C.mlirGoGlobalConstantOperationAddBody(unwrapOperation(op), unwrapBlock(body))
}

//===----------------------------------------------------------------------===//
// Casting Operations
//===----------------------------------------------------------------------===//

func NewBitcastOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateBitcastOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewComplexExtendOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateComplexExtendOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewComplexTruncateOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateComplexTruncateOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewIntToPtrOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateIntToPtrOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewPtrToIntOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreatePtrToIntOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewFloatTruncateOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateFloatTruncateOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewIntTruncateOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateIntTruncateOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewFloatExtendOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateFloatExtendOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewSignedExtendOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSignedExtendOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewZeroExtendOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateZeroExtendOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewFloatToUnsignedIntOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateFloatToUnsignedIntOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewFloatToSignedIntOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateFloatToSignedIntOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewUnsignedIntToFloatOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateUnsignedIntToFloatOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewSignedIntToFloatOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSignedIntToFloatOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewFunctionToPointerOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateFunctionToPointerOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewPointerToFunctionOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreatePointerToFunctionOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewChangeInterfaceOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateChangeInterfaceOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewTypeAssertOperation(ctx mlir.Context, value mlir.ValueLike, results []mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateTypeAssertOperation(
		unwrapContext(ctx),
		unwrapValue(value),
		C.int(len(results)),
		unwrapTypeSlice(results),
		unwrapLocation(location),
	))
}

func NewStringToSliceOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateStringToSliceOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

func NewSliceToStringOperation(ctx mlir.Context, value mlir.ValueLike, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateSliceToStringOperation(unwrapContext(ctx), unwrapValue(value), unwrapType(typ), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

func GetFunction(ctx mlir.Context, symbol string, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoGetFunction(unwrapContext(ctx), unwrapStringRef(mlir.NewStringRef(symbol)), unwrapType(typ), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Builtin Operations
//===----------------------------------------------------------------------===//

func NewPanicOperation(ctx mlir.Context, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreatePanicOperation(unwrapContext(ctx), unwrapValue(value), unwrapLocation(location)))
}

func NewRecoverOperation(ctx mlir.Context, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateRecoverOperation(unwrapContext(ctx), unwrapType(typ), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

func NewAtomicAddIOperation(ctx mlir.Context, resultType mlir.TypeLike, addr, delta mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAtomicAddIOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(addr), unwrapValue(delta), unwrapLocation(location)))
}

func NewAtomicCompareAndSwapOperation(ctx mlir.Context, resultType mlir.TypeLike, addr, old, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAtomicCompareAndSwapOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(addr), unwrapValue(old), unwrapValue(value), unwrapLocation(location)))
}

func NewAtomicSwapOperation(ctx mlir.Context, resultType mlir.TypeLike, addr, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateAtomicSwapOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(addr), unwrapValue(value), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

func NewBranchOperation(ctx mlir.Context, dest mlir.Block, destOperands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateBranchOperation(
		unwrapContext(ctx),
		unwrapBlock(dest),
		C.int(len(destOperands)),
		unwrapValueSlice(destOperands),
		unwrapLocation(location),
	))
}

func NewCondBranchOperation(ctx mlir.Context, condition mlir.ValueLike, trueDest mlir.Block, trueDestOperands []mlir.ValueLike, falseDest mlir.Block, falseDestOperands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCondBranchOperation(
		unwrapContext(ctx),
		unwrapValue(condition),
		unwrapBlock(trueDest),
		C.int(len(trueDestOperands)),
		unwrapValueSlice(trueDestOperands),
		unwrapBlock(falseDest),
		C.int(len(falseDestOperands)),
		unwrapValueSlice(falseDestOperands),
		unwrapLocation(location),
	))
}

func NewReturnOperation(ctx mlir.Context, operands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateReturnOperation(
		unwrapContext(ctx),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

//===----------------------------------------------------------------------===//
// Call Operations
//===----------------------------------------------------------------------===//

func NewCallOperation(ctx mlir.Context, callee string, resultTypes []mlir.TypeLike, operands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCallOperation(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(callee)),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

func NewClosureCallOperation(ctx mlir.Context, signature mlir.AttributeLike, callee mlir.ValueLike, resultTypes []mlir.TypeLike, operands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateClosureCallOperation(
		unwrapContext(ctx),
		unwrapAttribute(signature),
		unwrapValue(callee),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

func NewCallIndirectOperation(ctx mlir.Context, callee mlir.ValueLike, resultTypes []mlir.TypeLike, operands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateCallIndirectOperation(
		unwrapContext(ctx),
		unwrapValue(callee),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

func NewDeferOperation1(ctx mlir.Context, symName string, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDeferOperation1(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(symName)),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewDeferOperation2(ctx mlir.Context, symName mlir.AttributeLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDeferOperation2(
		unwrapContext(ctx),
		unwrapAttribute(symName),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewDeferOperation3(ctx mlir.Context, signature mlir.AttributeLike, calleeValue mlir.ValueLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDeferOperation3(
		unwrapContext(ctx),
		unwrapAttribute(signature),
		unwrapValue(calleeValue),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewDeferOperation4(ctx mlir.Context, ifaceValue mlir.ValueLike, methodName string, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDeferOperation4(
		unwrapContext(ctx),
		unwrapValue(ifaceValue),
		unwrapStringRef(mlir.NewStringRef(methodName)),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewDeferOperation5(ctx mlir.Context, ifaceValue mlir.ValueLike, methodName mlir.AttributeLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateDeferOperation5(
		unwrapContext(ctx),
		unwrapValue(ifaceValue),
		unwrapAttribute(methodName),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewGoOperation1(ctx mlir.Context, symName string, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGoOperation1(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(symName)),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewGoOperation2(ctx mlir.Context, symName mlir.AttributeLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGoOperation2(
		unwrapContext(ctx),
		unwrapAttribute(symName),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewGoOperation3(ctx mlir.Context, signature mlir.AttributeLike, calleeValue mlir.ValueLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGoOperation3(
		unwrapContext(ctx),
		unwrapAttribute(signature),
		unwrapValue(calleeValue),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewGoOperation4(ctx mlir.Context, ifaceValue mlir.ValueLike, methodName string, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGoOperation4(
		unwrapContext(ctx),
		unwrapValue(ifaceValue),
		unwrapStringRef(mlir.NewStringRef(methodName)),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewGoOperation5(ctx mlir.Context, ifaceValue mlir.ValueLike, methodName mlir.AttributeLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateGoOperation5(
		unwrapContext(ctx),
		unwrapValue(ifaceValue),
		unwrapAttribute(methodName),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewInterfaceCall(ctx mlir.Context, callee string, resultTypes []mlir.TypeLike, value mlir.ValueLike, args []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateInterfaceCall(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(callee)),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		unwrapValue(value),
		C.int(len(args)),
		unwrapValueSlice(args),
		unwrapLocation(location),
	))
}

func NewBuiltInCallOperation(ctx mlir.Context, identifier string, resultTypes []mlir.TypeLike, operands []mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateBuiltInCallOperation(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(identifier)),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		C.int(len(operands)),
		unwrapValueSlice(operands),
		unwrapLocation(location),
	))
}

//===----------------------------------------------------------------------===//
// Value Operations
//===----------------------------------------------------------------------===//

func NewZeroOperation(ctx mlir.Context, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateZeroOperation(unwrapContext(ctx), unwrapType(typ), unwrapLocation(location)))
}

func NewNilOperation(ctx mlir.Context, typ mlir.TypeLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateNilOperation(unwrapContext(ctx), unwrapType(typ), unwrapLocation(location)))
}

func NewComplexOperation(ctx mlir.Context, typ mlir.TypeLike, real, imag mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateComplexOperation(unwrapContext(ctx), unwrapType(typ), unwrapValue(real), unwrapValue(imag), unwrapLocation(location)))
}

func NewImagOperation(ctx mlir.Context, typ mlir.TypeLike, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateImagOperation(unwrapContext(ctx), unwrapType(typ), unwrapValue(value), unwrapLocation(location)))
}

func NewRealOperation(ctx mlir.Context, typ mlir.TypeLike, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateRealOperation(unwrapContext(ctx), unwrapType(typ), unwrapValue(value), unwrapLocation(location)))
}

func NewMakeMapOperation(ctx mlir.Context, resultType mlir.TypeLike, capacity mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMakeMapOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(capacity), unwrapLocation(location)))
}

func NewMakeSliceOperation(ctx mlir.Context, resultType mlir.TypeLike, length mlir.ValueLike, capacity mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMakeSliceOperation(unwrapContext(ctx), unwrapType(resultType), unwrapValue(length), unwrapValue(capacity), unwrapLocation(location)))
}

func NewMakeInterfaceOperation(ctx mlir.Context, resultType, typ mlir.TypeLike, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateMakeInterfaceOperation(unwrapContext(ctx), unwrapType(resultType), unwrapType(typ), unwrapValue(value), unwrapLocation(location)))
}

//===----------------------------------------------------------------------===//
// Channel Operations
//===----------------------------------------------------------------------===//

func NewChanRecvOp(ctx mlir.Context, resultTypes []mlir.TypeLike, channel mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateChanRecvOp(
		unwrapContext(ctx),
		C.int(len(resultTypes)),
		unwrapTypeSlice(resultTypes),
		unwrapValue(channel),
		unwrapLocation(location),
	))
}

func NewChanSendOp(ctx mlir.Context, channel, value mlir.ValueLike, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateChanSendOp(unwrapContext(ctx), unwrapValue(channel), unwrapValue(value), unwrapLocation(location)))
}

func NewChanRangeOp(ctx mlir.Context, channel mlir.ValueLike, bodyDest, exitDest mlir.Block, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateChanRangeOp(unwrapContext(ctx), unwrapValue(channel), unwrapBlock(bodyDest), unwrapBlock(exitDest), unwrapLocation(location)))
}

func unwrapBlockSlice(blocks []mlir.Block) *C.MlirBlock {
	if len(blocks) == 0 {
		return nil
	}
	rawBlocks := make([]C.MlirBlock, len(blocks))
	for i, b := range blocks {
		rawBlocks[i] = unwrapBlock(b)
	}
	return &rawBlocks[0]
}

func NewChanSelectOp(ctx mlir.Context, hasDefault bool, send mlir.AttributeLike, chans []mlir.ValueLike, defaultDest, exitDest mlir.Block, cases []mlir.Block, location mlir.LocationLike) mlir.Operation {
	return wrapOperation(C.mlirGoCreateChanSelectOp(
		unwrapContext(ctx),
		C.bool(hasDefault),
		unwrapAttribute(send),
		C.int(len(chans)),
		unwrapValueSlice(chans),
		unwrapBlock(defaultDest),
		unwrapBlock(exitDest),
		C.int(len(cases)),
		unwrapBlockSlice(cases),
		unwrapLocation(location),
	))
}
