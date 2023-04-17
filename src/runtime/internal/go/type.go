package _go

import "unsafe"

type typeDescriptor struct {
	name     unsafe.Pointer
	size     int64
	kind     int
	methods  unsafe.Pointer
	fields   unsafe.Pointer
	array    unsafe.Pointer
	mapp     unsafe.Pointer
	ptr      unsafe.Pointer
	channel  unsafe.Pointer
	function unsafe.Pointer
}

type methodTable struct {
	count   int
	methods unsafe.Pointer
}

type functionDescriptor struct {
	name        unsafe.Pointer
	argCount    int
	argTypes    unsafe.Pointer
	returnCount int
	returnTypes unsafe.Pointer
}

type fieldTable struct {
	count  int
	fields unsafe.Pointer
}

type fieldDescriptor struct {
	name      unsafe.Pointer
	fieldType unsafe.Pointer
	tagCount  int
	tags      unsafe.Pointer
}

type structTag struct {
	key   unsafe.Pointer
	value unsafe.Pointer
}

type arrayDescriptor struct {
	length      int64
	elementType unsafe.Pointer
	capacity    int64
}

type mapTypeDescriptor struct {
	keyType   unsafe.Pointer
	valueType unsafe.Pointer
}

type pointerDescriptor struct {
	elementType unsafe.Pointer
}

type channelTypeDescriptor struct {
	elementType unsafe.Pointer
	direction   int
}
