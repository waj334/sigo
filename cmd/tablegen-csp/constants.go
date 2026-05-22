package main

const (
	constRecordObject           = "object"
	constObjectFieldName        = "name"
	constObjectFieldDescription = "description"

	constRecordPeripheralType           = "PeripheralType"
	constPeripheralTypeFieldRegisters   = "registers"
	constPeripheralTypeFieldAccessWidth = "accessWidth"
	constPeripheralTypeFieldCount       = "count"
	constPeripheralTypeArrayLabel       = "arrayLabel"

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
	constMemoryRangeFieldAlign      = "align"

	constRecordAccessMode = "AccessMode"
	constAccessModeValue  = "value"

	constRecordVariant                  = "Variant"
	constVariantFieldMemories           = "memories"
	constVariantFieldInterrupts         = "interrupts"
	constVariantFieldStackSize          = "stackSize"
	constVariantFieldDefaultTextRegion  = "defaultTextRegion"
	constVariantFieldDefaultRAMRegion   = "defaultRAMRegion"
	constVariantFieldDefaultHeapRegion  = "defaultHeapRegion"
	constVariantFieldDefaultStackRegion = "defaultStackRegion"

	constRecordInterrupt    = "Interrupt"
	constInterruptFieldLine = "line"

	constRecordSeries               = "Series"
	constSeriesFieldArchitecture    = "arch"
	constSeriesFieldVariants        = "variants"
	constSeriesFieldPeripheralTypes = "peripheralTypes"
	constSeriesRuntimePackages      = "runtimePackages"

	constRecordArchitecture          = "Architecture"
	constArchitectureRuntimePackages = "runtimePackages"
	constArchitectureTags            = "tags"
)
