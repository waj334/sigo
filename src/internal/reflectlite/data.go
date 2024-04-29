package reflectlite

import "unsafe"

type _namedTypeData struct {
	underlyingType *_type
	methods        []_funcData
}

type _funcData struct {
	funcPtr   unsafe.Pointer
	signature *_type
}

type _signatureTypeData struct {
	id             uint32
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
	tag      *string
}

type _channelTypeData struct {
	elementType *_type
	direction   uint8
}

type _mapTypeData struct {
	keyType   *_type
	valueType *_type
}
