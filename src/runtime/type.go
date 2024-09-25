package runtime

import "unsafe"

type kind uint8

const (
	Invalid kind = iota
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	Array
	Chan
	Func
	Interface
	Map
	Pointer
	Slice
	String
	Struct
	UnsafePointer
)

type _type struct {
	kind kind
	size uint16
	data unsafe.Pointer
	name string
}

type _namedTypeData struct {
	underlyingType *_type
	methods        []*_funcData
}

type _funcData struct {
	id        uint32
	funcPtr   unsafe.Pointer
	signature *_type
}

type _interfaceData struct {
	methods []*_interfaceMethodData
}

type _interfaceMethodData struct {
	id        uint32
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
}

type _channelTypeData struct {
	elementType *_type
	direction   uint8
}

type _mapTypeData struct {
	keyType     *_type
	elementType *_type
}
