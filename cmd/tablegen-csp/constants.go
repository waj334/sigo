package main

const (
	constRecordObject           = "object"
	constObjectFieldName        = "name"
	constObjectFieldDescription = "description"

	constRecordPeripheralType           = "PeripheralType"
	constPeripheralTypeFieldRegisters   = "registers"
	constPeripheralTypeFieldAccessWidth = "accessWidth"

	constRecordPeripheralGroup         = "PeripheralGroup"
	constPeripheralGroupFieldInstances = "instances"

	constRecordPeripheralInstance    = "PeripheralInstance"
	constPeripheralInstanceFieldBase = "base"
	constPeripheralInstanceFieldType = "type"

	constRecordRegister      = "register"
	constRegisterFieldFields = "fields"
	constRegisterFieldCount  = "count"

	constRecordField      = "field"
	constFieldFieldAccess = "access"
	constFieldFieldEnums  = "enums"

	constRecordEnum     = "Enum"
	constEnumFieldValue = "value"

	constRecordRange      = "Range"
	constRecordBitRange   = "BitRange"
	constRecordByteRange  = "ByteRange"
	constRangeFieldOffset = "offset"
	constRangeFieldWidth  = "width"

	constRecordMemoryRange          = "MemoryRange"
	constMemoryRangeFieldAccess     = "access"
	constMemoryRangeFieldExecutable = "executable"

	constRecordAccessMode = "AccessMode"
	constAccessModeValue  = "value"

	constRecordVariant          = "Variant"
	constVariantFieldMemories   = "memories"
	constVariantFieldInterrupts = "interrupts"
	constVariantFieldStackSize  = "stackSize"

	constRecordInterrupt    = "Interrupt"
	constInterruptFieldLine = "line"
)
