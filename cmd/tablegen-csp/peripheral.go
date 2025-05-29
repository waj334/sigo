package main

import (
	"fmt"
	"go/format"
	"io"
	"os"
	"strings"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generatePeripheralType(out io.Writer, peripheralType *tablegen.Record, instances []*tablegen.Record, groups []*tablegen.Record, requiredTags []string, optionalTags []string) (int, error) {
	if len(instances) == 0 {
		return 0, nil
	}

	var builder strings.Builder

	peripheralName := peripheralType.GetValueAsString(constObjectFieldName)
	peripheralCount := peripheralType.GetValueAsInt(constPeripheralTypeFieldCount)

	packageName := strings.ToLower(formatGoIdentifier(strings.ToLower(peripheralName), true))
	className := "_" + formatGoIdentifier(strings.ToLower(peripheralName), false)
	regsClassName := "_regs" + formatGoIdentifier(strings.ToLower(peripheralName), true)

	if len(requiredTags) > 0 || len(optionalTags) > 0 {
		fmt.Fprintf(&builder, "//go:build %s\n\n", tagsString(requiredTags, optionalTags))
	}

	fmt.Fprintf(&builder, "package %s\n\n", packageName)
	fmt.Fprintf(&builder, "import (\n")
	fmt.Fprintf(&builder, "\"unsafe\"\n")
	fmt.Fprintf(&builder, "\"volatile\"\n")
	fmt.Fprintf(&builder, ")\n")
	fmt.Fprintf(&builder, "\n")
	fmt.Fprintf(&builder, "var (\n")
	for _, instance := range instances {
		instanceName := instance.GetValueAsString(constObjectFieldName)
		varName := formatGoIdentifier(strings.ToLower(instanceName), true)
		baseAddress := instance.GetValueAsInt(constPeripheralInstanceFieldBase)

		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n", varName, className, baseAddress)
	}

	if len(groups) > 0 {
		fmt.Fprintf(&builder, "\n")
		for _, group := range groups {
			varName := formatGoIdentifier(group.GetValueAsString(constObjectFieldName), true)
			members := group.GetValueAsListOfDefs(constPeripheralGroupFieldInstances)
			fmt.Fprintf(&builder, "%s = [%d]*%s{\n", varName, len(members), className)
			for _, member := range members {
				instanceName := member.GetValueAsString(constObjectFieldName)
				memberVarName := formatGoIdentifier(strings.ToLower(instanceName), true)
				fmt.Fprintf(&builder, "%s,\n", memberVarName)
			}
			fmt.Fprintf(&builder, "}\n")
		}
	}
	fmt.Fprintf(&builder, ")\n")
	fmt.Fprintf(&builder, "\n")
	fmt.Fprintf(&builder, "type %s struct {\n", className)

	if peripheralCount > 1 {
		memberName := formatGoIdentifier(peripheralName, true)
		arrayLabel := peripheralType.GetValueAsString(constPeripheralTypeArrayLabel)
		if len(arrayLabel) == 0 {
			arrayLabel = memberName
		}
		arrayLabel = formatGoIdentifier(arrayLabel, true)

		fmt.Fprintf(&builder, "%s [%d]%s\n", arrayLabel, peripheralCount, regsClassName)
		fmt.Fprintf(&builder, "}\n\n")
		fmt.Fprintf(&builder, "type %s struct {\n", regsClassName)
	}

	// Generate struct members.
	position := int64(0)
	registers := peripheralType.GetValueAsListOfDefs(constPeripheralTypeFieldRegisters)
	for _, register := range registers {
		offset := register.GetValueAsInt(constRangeFieldOffset)
		count := register.GetValueAsInt(constRegisterFieldCount)
		width := (register.GetValueAsInt(constRangeFieldWidth) / 8) * count

		// Insert padding if currentOffset < register's offset
		if offset > position {
			padding := offset - position
			fmt.Fprintf(&builder, "_ [%d]byte\n", padding)
		}

		fieldName := formatRegisterFieldName(register)
		typeName := formatRegisterTypeName(register)

		if count > 1 {
			fmt.Fprintf(&builder, "%s [%d]%s\n", fieldName, count, typeName)
		} else {
			fmt.Fprintf(&builder, "%s %s\n", fieldName, typeName)
		}

		// Update position.
		position = offset + width
	}

	fmt.Fprintf(&builder, "}\n\n")

	// Generate register types.
	for _, register := range registers {
		_, err := generateRegister(&builder, register)
		if err != nil {
			return 0, err
		}
	}

	// Format the final output.
	srcStr := builder.String()
	src, err := format.Source([]byte(srcStr))
	if err != nil {
		fmt.Fprintf(os.Stderr, "*** START ***\n%s\n*** END ***\n", srcStr)
		return 0, err
	}
	return fmt.Fprint(out, string(src))
}

func generateRegister(out io.Writer, register *tablegen.Record) (int, error) {
	var builder strings.Builder

	registerWidth := register.GetValueAsInt(constRangeFieldWidth)
	registerTypeName := formatRegisterTypeName(register)
	registerDescription := register.GetValueAsString(constObjectFieldDescription)
	registerUnderlyingType, err := typeForWidth(registerWidth)
	if err != nil {
		return 0, err
	}

	if len(registerDescription) > 0 {
		fmt.Fprintf(&builder, "// %s %s\n", registerTypeName, registerDescription)
	}

	fmt.Fprintf(&builder, "type %s %s\n\n", registerTypeName, registerUnderlyingType)

	fields := register.GetValueAsListOfDefs(constRegisterFieldFields)
	for _, field := range fields {
		fieldName := formatRegisterFieldName(field)
		fieldDescription := field.GetValueAsString(constObjectFieldDescription)
		fieldAccess := field.GetValueAsDef(constFieldFieldAccess).GetValueAsString(constAccessModeValue)
		fieldOffset := field.GetValueAsInt(constRangeFieldOffset)
		fieldWidth := field.GetValueAsInt(constRangeFieldWidth)
		fieldEnums := field.GetValueAsListOfDefs(constFieldFieldEnums)
		fieldEnumType := formatRegisterFieldEnumTypeName(register, field)
		fieldUnderlyingType, err := typeForWidth(fieldWidth)
		if err != nil {
			return 0, err
		}

		// Generate constants.
		constPrefix := formatRegisterConstPrefix(register, field)
		constShift := fmt.Sprintf("%sShift", constPrefix)
		constMask := fmt.Sprintf("%sMask", constPrefix)
		mask := ((1 << fieldWidth) - 1) << fieldOffset

		if len(fieldEnums) > 0 {
			fmt.Fprintf(&builder, "type %s %s\n\n", fieldEnumType, fieldUnderlyingType)
		}

		fmt.Fprintf(&builder, "const (\n")

		setterParamType := fieldUnderlyingType
		if len(fieldEnums) > 0 {
			for _, enum := range fieldEnums {
				setterParamType = fieldEnumType
				enumName := formatRegisterFieldEnumValueName(register, field, enum)
				enumDescription := enum.GetValueAsString(constObjectFieldDescription)
				enumValue := enum.GetValueAsInt(constEnumFieldValue)

				trailingNewline := false
				if len(enumDescription) > 0 {
					fmt.Fprintf(&builder, "// %s %s\n", enumName, enumDescription)
					trailingNewline = true
				}

				if fieldWidth == 1 {
					if enumValue == 0 {
						fmt.Fprintf(&builder, "%s %s = false\n", enumName, fieldEnumType)
					} else {
						fmt.Fprintf(&builder, "%s %s = true\n", enumName, fieldEnumType)
					}
				} else {
					fmt.Fprintf(&builder, "%s %s = %#x\n", enumName, fieldEnumType, enumValue)
				}

				if trailingNewline {
					fmt.Fprintf(&builder, "\n")
				}
			}
			fmt.Fprintf(&builder, "\n")
		} else if fieldWidth == 1 {
			setterParamType = "bool"
		}

		fmt.Fprintf(&builder, "%s = %d\n", constShift, fieldOffset)
		fmt.Fprintf(&builder, "%s = %#x\n", constMask, mask)
		fmt.Fprintf(&builder, ")\n\n")

		addr := fmt.Sprintf("(*%s)(r)", registerUnderlyingType)
		load := fmt.Sprintf("volatile.LoadUint%d(%s)", registerWidth, addr)

		if strings.Contains(fieldAccess, "R") {
			// Generate read API.
			if len(fieldDescription) > 0 {
				fmt.Fprintf(&builder, "// Get%s %s\n", fieldName, fieldDescription)
			}

			if fieldWidth == 1 {
				fmt.Fprintf(&builder, "func (r *%s) Get%s() bool {\n", registerTypeName, fieldName)
				fmt.Fprintf(&builder, "return (%s&%s) != 0\n", load, constMask)
				fmt.Fprintf(&builder, "}\n\n")
			} else {
				fmt.Fprintf(&builder, "func (r *%s) Get%s() %s {\n", registerTypeName, fieldName, fieldUnderlyingType)
				fmt.Fprintf(&builder, "return %s((%s&%s) >> %s)\n", fieldUnderlyingType, load, constMask, constShift)
				fmt.Fprintf(&builder, "}\n\n")
			}
		}

		if strings.Contains(fieldAccess, "W") {
			// Generate write API.
			if len(fieldDescription) > 0 {
				fmt.Fprintf(&builder, "// Set%s %s\n", fieldName, fieldDescription)
			}

			if fieldWidth == 1 {
				fmt.Fprintf(&builder, "func (r *%s) Set%s(value %s) {\n", registerTypeName, fieldName, setterParamType)
				fmt.Fprintf(&builder, "if value {\n")
				fmt.Fprintf(&builder, "volatile.StoreUint%d(%s, %s|%s)\n", registerWidth, addr, load, constMask)
				fmt.Fprintf(&builder, "} else {\n")
				fmt.Fprintf(&builder, "volatile.StoreUint%d(%s, %s&^%s)\n", registerWidth, addr, load, constMask)
				fmt.Fprintf(&builder, "}\n")
				fmt.Fprintf(&builder, "}\n\n")
			} else {
				fmt.Fprintf(&builder, "func (r *%s) Set%s(value %s) {\n", registerTypeName, fieldName, setterParamType)
				fmt.Fprintf(&builder, "volatile.StoreUint%d(%s, (%s&^%s)|(%s(value)<<%s))\n", registerWidth, addr, load, constMask, registerUnderlyingType, constShift)
				fmt.Fprintf(&builder, "}\n\n")
			}
		}
	}

	return fmt.Fprint(out, builder.String())
}

func formatRegisterConstPrefix(register, field *tablegen.Record) string {
	registerName := register.GetValueAsString(constObjectFieldName)
	registerName = formatCamelCase(strings.Split(registerName, "_")...)

	fieldName := field.GetValueAsString(constObjectFieldName)
	fieldName = formatCamelCase(strings.Split(fieldName, "_")...)
	return formatGoIdentifier(formatCamelCase("Register", registerName, "Field", fieldName), true)
}

func formatRegisterFieldName(def *tablegen.Record) string {
	fieldName := def.GetValueAsString(constObjectFieldName)
	fieldName = formatCamelCase(strings.Split(fieldName, "_")...)
	return formatGoIdentifier(strings.ToLower(fieldName), true)
}

func formatRegisterTypeName(def *tablegen.Record) string {
	registerName := def.GetValueAsString(constObjectFieldName)
	registerName = formatCamelCase(strings.Split(registerName, "_")...)

	return formatGoIdentifier(formatCamelCase("register", registerName, "Type"), false)
}

func formatRegisterFieldEnumTypeName(register, field *tablegen.Record) string {
	registerName := register.GetValueAsString(constObjectFieldName)
	registerName = formatCamelCase(strings.Split(registerName, "_")...)

	fieldName := field.GetValueAsString(constObjectFieldName)
	return formatGoIdentifier(formatCamelCase("Register", registerName, "Field", fieldName, "Enum", "Type"), true)
}

func formatRegisterFieldEnumValueName(register, field, enum *tablegen.Record) string {
	registerName := register.GetValueAsString(constObjectFieldName)
	registerName = formatCamelCase(strings.Split(registerName, "_")...)

	fieldName := field.GetValueAsString(constObjectFieldName)
	fieldName = formatCamelCase(strings.Split(fieldName, "_")...)
	enumName := enum.GetValueAsString(constObjectFieldName)
	return formatGoIdentifier(formatCamelCase("Register", registerName, "Field", fieldName, "Enum", enumName), true)
}
