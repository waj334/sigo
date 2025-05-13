package main

import (
	"fmt"
	"go/format"
	"io"
	"strings"

	"omibyte.io/sigo/llvm/tablegen"
)

func generatePeripheralType(out io.Writer, peripheralType tablegen.Record, instances []tablegen.Record, groups []tablegen.Record) (int, error) {
	if len(instances) == 0 {
		return 0, nil
	}

	var builder strings.Builder

	peripheralName := peripheralType.GetValueAsString(constObjectFieldName)
	packageName := formatGoIdentifier(strings.ToLower(peripheralName), true)
	className := formatGoIdentifier(strings.ToLower(peripheralName), false)

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

	// Generate struct members.
	position := int64(0)
	registers := peripheralType.GetValueAsListOfDefs(constPeripheralTypeFieldRegisters)
	for _, register := range registers {
		offset := register.GetValueAsInt(constRangeFieldOffset)
		width := register.GetValueAsInt(constRangeFieldWidth) / 8

		// Insert padding if currentOffset < register's offset
		if offset > position {
			padding := offset - position
			fmt.Fprintf(&builder, "_ [%d]byte\n", padding)
		}

		fieldName := formatRegisterFieldName(register)
		typeName := formatRegisterTypeName(register)
		fmt.Fprintf(&builder, "%s %s\n", fieldName, typeName)

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
		return 0, err
	}
	return fmt.Fprint(out, string(src))
}

func generateRegister(out io.Writer, register tablegen.Record) (int, error) {
	var builder strings.Builder

	registerWidth := register.GetValueAsInt(constRangeFieldWidth)
	registerTypeName := formatRegisterTypeName(register)
	underlyingType, err := typeForWidth(registerWidth)
	if err != nil {
		return 0, err
	}

	fmt.Fprintf(&builder, "type %s %s\n\n", registerTypeName, underlyingType)

	fields := register.GetValueAsListOfDefs(constRegisterFieldFields)
	for _, field := range fields {
		fieldName := formatRegisterFieldName(field)
		fieldAccess := field.GetValueAsDef(constFieldFieldAccess).GetValueAsString(constAccessModeValue)
		fieldWidth := field.GetValueAsInt(constRangeFieldWidth)
		fieldType, err := typeForWidth(fieldWidth)
		if err != nil {
			return 0, err
		}

		if strings.Contains(fieldAccess, "R") {
			// Generate read API.
			fmt.Fprintf(&builder, "func (r *%s) Get%s() %s {\n", registerTypeName, fieldName, fieldType)
			fmt.Fprintf(&builder, "// TODO: Implement me\n")
			fmt.Fprintf(&builder, "return 0")
			fmt.Fprintf(&builder, "}\n\n")
		}

		if strings.Contains(fieldAccess, "W") {
			// TODO: Generate write API.
			fmt.Fprintf(&builder, "func (r *%s) Set%s(value %s) {\n", registerTypeName, fieldName, fieldType)
			fmt.Fprintf(&builder, "// TODO: Implement me\n")
			fmt.Fprintf(&builder, "}\n\n")
		}
	}

	return fmt.Fprint(out, builder.String())
}

func formatRegisterFieldName(def tablegen.Record) string {
	name := def.GetValueAsString(constObjectFieldName)
	return formatGoIdentifier(strings.ToLower(name), true)
}

func formatRegisterTypeName(def tablegen.Record) string {
	name := def.GetValueAsString(constObjectFieldName)
	return formatGoIdentifier(formatCamelCase("register", name, "Type"), false)
}
