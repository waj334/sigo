package reflectlite

import "unsafe"

// All structures below must match the corresponding definitions in
// runtime/type.go exactly (field order, types, and sizes).

type _namedTypeData struct {
	underlyingType *_type
	methods        []*_funcData
}

type _funcData struct {
	id        uint32
	funcPtr   unsafe.Pointer
	signature *_signatureTypeData
}

type _signatureTypeData struct {
	receiverType   *_type
	parameterTypes []*_type
	returnTypes    []*_type
}

type _arrayTypeData struct {
	length      uint16
	elementType *_type
}

type _structTypeData struct {
	fields []_structFieldData
}

type _structFieldData struct {
	dataType *_type
	tag      string
	offset   uintptr
}

type _channelTypeData struct {
	elementType *_type
	direction   uint8
}

type _mapTypeData struct {
	keyType     *_type
	elementType *_type
}

type _interfaceData struct {
	methods []*_interfaceMethodData
}

type _interfaceMethodData struct {
	id        uint32
	signature *_signatureTypeData
}
