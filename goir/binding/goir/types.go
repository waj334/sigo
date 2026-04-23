package goir

/*
#include <Go-c/mlir/Types.h>
#include <stdlib.h>
*/
import "C"
import (
	"pkg.si-go.dev/go-mlir/mlir"
)

type UntypedType struct {
	mlir.Type
}

func NewUntypedType(ctx mlir.Context, basicType BasicType) UntypedType {
	return UntypedType{wrapType(C.mlirGoCreateUntypedType(unwrapContext(ctx), C.enum_mlirGoBasicType(basicType)))}
}

func AsUntypedType(T mlir.TypeLike) (UntypedType, bool) {
	if C.mlirGoTypeIsUntyped(unwrapType(T)) {
		return UntypedType{wrapType(unwrapType(T))}, true
	}
	return UntypedType{}, false
}

func (t UntypedType) BasicKind() BasicType {
	return BasicType(C.mlirGoUntypedTypeGetHBasicKind(unwrapType(t)))
}

type NamedType struct {
	mlir.Type
}

func NewNamedType(underlying mlir.TypeLike, name string, methods mlir.AttributeLike) NamedType {
	return NamedType{wrapType(C.mlirGoCreateNamedType(unwrapType(underlying), unwrapStringRef(mlir.NewStringRef(name)), unwrapAttribute(methods)))}
}

func GetUnderlyingType(t mlir.TypeLike) mlir.Type {
	return wrapType(C.mlirGoGetUnderlyingType(unwrapType(t)))
}

func GetBaseType(t mlir.TypeLike) mlir.Type {
	return wrapType(C.mlirGoGetBaseType(unwrapType(t)))
}

func TypeIsAPointer(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsAPointer(unwrapType(t)))
}

func TypeIsAInterface(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsAInterface(unwrapType(t)))
}

type ArrayType struct {
	mlir.Type
}

func NewArrayType(elementType mlir.TypeLike, length int) ArrayType {
	return ArrayType{wrapType(C.mlirGoCreateArrayType(unwrapType(elementType), C.int(length)))}
}

type ChanType struct {
	mlir.Type
}

func NewChanType(elementType mlir.TypeLike, direction ChanDirection) ChanType {
	return ChanType{wrapType(C.mlirGoCreateChanType(unwrapType(elementType), C.enum_mlirGoChanDirection(direction)))}
}

type InterfaceType struct {
	mlir.Type
}

func NewInterfaceType(ctx mlir.Context, methodNames []string, methods []FunctionType) InterfaceType {
	return InterfaceType{wrapType(C.mlirGoCreateInterfaceType(
		unwrapContext(ctx),
		C.int(len(methodNames)),
		unwrapStringSlice(methodNames),
		C.int(len(methods)),
		unwrapTypeSlice(methods),
	))}
}

func NewNamedInterfaceType(ctx mlir.Context, name string) InterfaceType {
	return InterfaceType{wrapType(C.mlirGoCreateNamedInterfaceType(unwrapContext(ctx), unwrapStringRef(mlir.NewStringRef(name))))}
}

func SetNamedInterfaceMethods(ctx mlir.Context, iface InterfaceType, methodNames []string, methods []FunctionType) {
	C.mlirGoSetNamedInterfaceMethods(
		unwrapContext(ctx),
		unwrapType(iface),
		C.int(len(methodNames)),
		unwrapStringSlice(methodNames),
		C.int(len(methods)),
		unwrapTypeSlice(methods),
	)
}

type MapType struct {
	mlir.Type
}

func NewMapType(keyType, valueType mlir.TypeLike) MapType {
	return MapType{wrapType(C.mlirGoCreateMapType(unwrapType(keyType), unwrapType(valueType)))}
}

type PointerType struct {
	mlir.Type
}

func NewPointerType(elementType mlir.TypeLike) PointerType {
	return PointerType{wrapType(C.mlirGoCreatePointerType(unwrapType(elementType)))}
}

func AsPointerType(typ mlir.TypeLike) (PointerType, bool) {
	if C.mlirGoTypeIsAPointer(unwrapType(typ)) {
		return PointerType{wrapType(unwrapType(typ))}, true
	}
	return PointerType{}, false
}

func (t PointerType) ElementType() mlir.Type {
	return wrapType(C.mlirGoPointerTypeGetElementType(unwrapType(t)))
}

func (t PointerType) SetElementType(elementType mlir.TypeLike) {
	C.mlirGoSetPointerElementType(unwrapType(t), unwrapType(elementType))
}

type UnsafePointerType struct {
	mlir.Type
}

func NewUnsafePointerType(ctx mlir.Context) UnsafePointerType {
	return UnsafePointerType{wrapType(C.mlirGoCreateUnsafePointerType(unwrapContext(ctx)))}
}

func NewDeferredPointerType(ctx mlir.Context, id string) PointerType {
	return PointerType{
		wrapType(C.mlirGoCreateDeferredPointerType(
			unwrapContext(ctx),
			unwrapStringRef(mlir.NewStringRef(id)))),
	}
}

type SliceType struct {
	mlir.Type
}

func NewSliceType(elementType mlir.TypeLike) SliceType {
	return SliceType{wrapType(C.mlirGoCreateSliceType(unwrapType(elementType)))}
}

type StringType struct {
	mlir.Type
}

func NewStringType(ctx mlir.Context) StringType {
	return StringType{wrapType(C.mlirGoCreateStringType(unwrapContext(ctx)))}
}

type StructType struct {
	mlir.Type
}

func NewBasicStructType(ctx mlir.Context, fields []mlir.TypeLike) StructType {
	return StructType{wrapType(C.mlirGoCreateBasicStructType(
		unwrapContext(ctx),
		C.int(len(fields)),
		unwrapTypeSlice(fields),
	))}
}

func NewLiteralStructType(ctx mlir.Context, names []mlir.StringAttr, fields []mlir.TypeLike, tags []mlir.StringAttr) StructType {
	return StructType{wrapType(C.mlirGoCreateLiteralStructType(
		unwrapContext(ctx),
		C.int(len(names)),
		unwrapAttributeSlice(names),
		C.int(len(fields)),
		unwrapTypeSlice(fields),
		C.int(len(tags)),
		unwrapAttributeSlice(tags),
	))}
}

func NewNamedStructType(ctx mlir.Context, name string) StructType {
	return StructType{wrapType(C.mlirGoCreateNamedStructType(unwrapContext(ctx), unwrapStringRef(mlir.NewStringRef(name))))}
}

func SetStructTypeBody(t StructType, names []mlir.StringAttr, fields []mlir.TypeLike, tags []mlir.StringAttr) {
	C.mlirGoSetStructTypeBody(
		unwrapType(t),
		C.int(len(names)),
		unwrapAttributeSlice(names),
		C.int(len(fields)),
		unwrapTypeSlice(fields),
		C.int(len(tags)),
		unwrapAttributeSlice(tags),
	)
}

func (t StructType) FieldType(index int) mlir.Type {
	return wrapType(C.mlirGoStructTypeGetFieldType(unwrapType(t), C.int(index)))
}

type BooleanType struct {
	mlir.Type
}

func NewBooleanType(ctx mlir.Context) BooleanType {
	return BooleanType{wrapType(C.mlirGoCreateBooleanType(unwrapContext(ctx)))}
}

func TypeIsBoolean(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsBoolean(unwrapType(t)))
}

type IntegerType struct {
	mlir.Type
}

func NewSignedIntType(ctx mlir.Context, width int) IntegerType {
	return IntegerType{wrapType(C.mlirGoCreateSignedIntType(unwrapContext(ctx), C.int(width)))}
}

func NewUnsignedIntType(ctx mlir.Context, width int) IntegerType {
	return IntegerType{wrapType(C.mlirGoCreateUnsignedIntType(unwrapContext(ctx), C.int(width)))}
}

func NewUintptrType(ctx mlir.Context) IntegerType {
	return IntegerType{wrapType(C.mlirGoCreateUintptrType(unwrapContext(ctx)))}
}

func AsIntegerType(typ mlir.TypeLike) (IntegerType, bool) {
	if C.mlirGoTypeIsInteger(unwrapType(typ)) {
		return IntegerType{wrapType(unwrapType(typ))}, true
	}
	return IntegerType{}, false
}

func TypeIsInteger(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsInteger(unwrapType(t)))
}

func TypeIsUntyped(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsUntyped(unwrapType(t)))
}

func (t IntegerType) IsSigned() bool {
	return bool(C.mlirGoIntegerTypeIsSigned(unwrapType(t)))
}

func (t IntegerType) IsUnsigned() bool {
	return bool(C.mlirGoIntegerTypeIsUnsigned(unwrapType(t)))
}

func (t IntegerType) IsUintptr() bool {
	return bool(C.mlirGoIntegerTypeIsUintptr(unwrapType(t)))
}

func (t IntegerType) Width() int {
	return int(C.mlirGoIntegerTypeGetWidth(unwrapType(t)))
}

type FunctionType struct {
	mlir.Type
}

func NewFunctionType(ctx mlir.Context, receiver mlir.TypeLike, inputs, results []mlir.TypeLike) FunctionType {
	return FunctionType{wrapType(C.mlirGoCreateFunctionType(
		unwrapContext(ctx),
		unwrapType(receiver),
		C.int(len(inputs)),
		unwrapTypeSlice(inputs),
		C.int(len(results)),
		unwrapTypeSlice(results),
	))}
}

func TypeIsAFunctionType(t mlir.TypeLike) bool {
	return bool(C.mlirGoTypeIsAFunctionType(unwrapType(t)))
}

func (t FunctionType) HasReceiver() bool {
	return bool(C.mlirGoFunctionTypeHasReceiver(unwrapType(t)))
}

func (t FunctionType) Receiver() mlir.Type {
	return wrapType(C.mlirGoFunctionTypeGetReceiver(unwrapType(t)))
}

func (t FunctionType) NumInputs() int {
	return int(C.mlirGoFunctionTypeGetNumInputs(unwrapType(t)))
}

func (t FunctionType) Input(index int) mlir.Type {
	return wrapType(C.mlirGoFunctionTypeGetInput(unwrapType(t), C.int(index)))
}

func (t FunctionType) NumResults() int {
	return int(C.mlirGoFunctionTypeGetNumResults(unwrapType(t)))
}

func (t FunctionType) Result(index int) mlir.Type {
	return wrapType(C.mlirGoFunctionTypeGetResult(unwrapType(t), C.int(index)))
}
