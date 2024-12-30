package main

import (
	"fmt"
	"go/format"
	"io"
	"slices"
	"strings"

	"omibyte.io/sigo/targets/device"
)

func writePeripheralsApi(output io.StringWriter, peripheral device.Peripheral) (int, error) {
	var builder strings.Builder

	fmt.Fprintf(&builder, "package %s\n\n", formatSymbol(peripheral.Identifier, false))

	builder.WriteString(`import (
	"unsafe"
	"volatile"
)

`)

	builder.WriteString("var (\n")
	writePeripheralConstants(&builder, peripheral)
	builder.WriteString(")\n")
	builder.WriteString("\n")

	writePeripheralTypeDeclaration(&builder, peripheral)
	builder.WriteString("\n\n")
	for _, registerGroup := range peripheral.RegisterGroups {
		writeRegisterGroupTypeDeclaration(&builder, registerGroup)
		for _, register := range registerGroup.Registers {
			writeRegisterTypeDeclaration(&builder, register)
			builder.WriteString("\n\n")

			writeRegisterMethods(&builder, register)
		}
	}

	srcStr := builder.String()
	src, err := format.Source([]byte(srcStr))
	if err != nil {
		panic(err)
	}

	return output.WriteString(string(src))
}

func writePeripheralTypeDeclaration(output io.StringWriter, p device.Peripheral) (int, error) {
	var builder strings.Builder

	if len(p.Description) > 0 {
		fmt.Fprintf(&builder, "// %s %s\n", typeName(p), p.Description)
	}

	excluded := map[string]struct{}{}

	// Visit each top-level group and find any subgroup that refers to a top-level group. These will NOT be instantiated
	// at the top of the struct directly.
	for _, group := range p.RegisterGroups {
		for _, subgroup := range group.Groups {
			if len(subgroup.Reference) > 0 {
				excluded[subgroup.Reference] = struct{}{}
			}
		}
	}

	var groups []device.RegisterGroup
	for _, group := range p.RegisterGroups {
		if _, ok := excluded[group.Identifier]; ok {
			continue
		}
		groups = append(groups, group)
	}

	fmt.Fprintf(&builder, "type %[1]s struct {\n", typeName(p))

	merge := false
	if len(groups) == 1 {
		if groups[0].Count == 1 {
			merge = true
		}
	}

	if merge {
		// Embed the register group type.
		fmt.Fprintf(&builder, "%[1]s\n", typeName(groups[0]))
	} else {

		// Sort the groups by offset.
		slices.SortFunc(groups, func(a, b device.RegisterGroup) int {
			return int(a.Offset - b.Offset)
		})

		offset := groups[0].Offset

		for i, group := range groups {

			if i > 0 {
				// Calculate padding required if there is a gap
				padding := group.Offset - (groups[i-1].Offset + groups[i-1].TotalWidth()/8)
				if padding > 0 {
					// Write padding bytes
					fmt.Fprintf(&builder, "_ [%d]uint8\n", padding)
					offset += padding
				}
			}

			if group.Count == 1 {
				fmt.Fprintf(&builder, "%s %s\n", varName(group), typeName(group))
			} else {
				fmt.Fprintf(&builder, "%s [%d]%s\n", varName(group), group.Count, typeName(group))
			}
			offset += group.TotalWidth() / 8
		}
	}

	fmt.Fprintln(&builder, "}")

	return output.WriteString(builder.String())
}

func writePeripheralConstants(output io.StringWriter, p device.Peripheral) (int, error) {
	var builder strings.Builder
	if len(p.Instances) == 0 && p.BaseAddress != nil {
		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n",
			varName(p), typeName(p), *p.BaseAddress)
	} else if len(p.Instances) == 1 {
		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n",
			varName(p), typeName(p), p.Instances[0])
	} else {
		fmt.Fprintf(&builder, "%s = [%d]*%s{\n", varName(p), len(p.Instances), typeName(p))
		for _, instance := range p.Instances {
			fmt.Fprintf(&builder, "(*%s)(unsafe.Pointer(uintptr(%#x))),\n", typeName(p), instance)
		}
		fmt.Fprintf(&builder, "}\n")
	}
	return output.WriteString(builder.String())
}

func writeRegisterGroupTypeDeclaration(output io.StringWriter, group device.RegisterGroup) (int, error) {
	var builder strings.Builder

	if len(group.Description) > 0 {
		fmt.Fprintf(&builder, "// %s %s\n", typeName(group), group.Description)
	}

	fmt.Fprintf(&builder, "type %[1]s struct {\n", typeName(group))
	if len(group.Groups) > 0 {
		offset := group.Groups[0].Offset
		for i, subgroup := range group.Groups {

			if i > 0 {
				// Calculate padding required if there is a gap
				padding := subgroup.Offset - (group.Groups[i-1].Offset + group.Groups[i-1].TotalWidth()/8)
				if padding > 0 {
					// Write padding bytes
					fmt.Fprintf(&builder, "_ [%d]uint8\n", padding)
					offset += padding
				}
			}

			// TODO: Need to look up the actual register group within the module!

			if subgroup.Count == 1 {
				fmt.Fprintf(&builder, "%s %s\n", varName(subgroup), typeName(subgroup))
			} else {
				fmt.Fprintf(&builder, "%s [%d]%s\n", varName(subgroup), subgroup.Count, typeName(subgroup))
			}
			offset += subgroup.TotalWidth() / 8
		}

		if offset < group.TotalWidth() {
			// Add tail padding.
			fmt.Fprintf(&builder, "_ [%d]uint8\n", group.TotalWidth()-offset)
		}
	} else if len(group.Registers) > 0 {
		offset := group.Registers[0].Offset
		for i, r := range group.Registers {
			if i > 0 {
				// Calculate padding required if there is a gap
				padding := r.Offset - (group.Registers[i-1].Offset + group.Registers[i-1].TotalWidth()/8)
				if padding > 0 {
					// Write padding bytes
					fmt.Fprintf(&builder, "_ [%d]uint8\n", padding)
					offset += padding
				}
			}

			offset += r.TotalWidth() / 8

			if r.Count > 1 {
				fmt.Fprintf(&builder, "%s [%d]%s\n", varName(r), r.Count, typeName(r))
			} else {
				fmt.Fprintf(&builder, "%s %s\n", varName(r), typeName(r))
			}
		}

		if offset < group.TotalWidth() {
			// Add tail padding.
			fmt.Fprintf(&builder, "_ [%d]uint8\n", group.TotalWidth()-offset)
		}
	}
	fmt.Fprintln(&builder, "}")

	return output.WriteString(builder.String())
}

func writeRegisterTypeDeclaration(output io.StringWriter, r device.Register) (int, error) {
	var builder strings.Builder

	if len(r.Description) > 0 {
		fmt.Fprintf(&builder, "// %s %s\n", typeName(r), r.Description)
	}

	fmt.Fprintf(&builder, "type %s uint%d", typeName(r), r.Width)

	return output.WriteString(builder.String())
}

func writeRegisterMethods(output io.StringWriter, r device.Register) (int, error) {
	var builder strings.Builder
	for _, f := range r.Fields {
		if f.Constants != nil {
			writeConstantTypeDeclaration(&builder, *f.Constants)
			builder.WriteString("\n")

			writeConstantValues(&builder, *f.Constants)
			builder.WriteString("\n")
		}

		writeFieldMethods(&builder, f)
	}
	return output.WriteString(builder.String())
}

func writeConstantTypeDeclaration(output io.Writer, c device.ConstantGroup) (int, error) {
	return fmt.Fprintf(output, "type %s %s", typeName(c), device.DataType(c.Field().Width))
}

func writeConstantValues(output io.StringWriter, c device.ConstantGroup) (int, error) {
	var builder strings.Builder

	fmt.Fprintf(&builder, "const (\n")
	for _, value := range c.Values {
		var v any = value.Value
		if device.DataType(c.Field().Width) == "bool" {
			v = value.Value == 1
		}
		fmt.Fprintf(&builder, "%s %s = %v", varName(value), typeName(c), v)
		if len(value.Description) > 0 {
			fmt.Fprintf(&builder, " // %s", value.Description)
		}
		builder.WriteString("\n")
	}
	fmt.Fprintf(&builder, ")\n")

	return output.WriteString(builder.String())
}

func writeFieldMethods(output io.StringWriter, f device.Field) (int, error) {
	var builder strings.Builder
	if f.Flags.IsSet(device.Read) {
		if f.Width == 1 {
			// Output an API that uses the `bool` data type  as the return type.
			builder.WriteString(boolGetter(f))
		} else {
			// Output an API that uses an integer data type of the necessary bit width as the return type.
			builder.WriteString(intGetter(f))
		}
		builder.WriteString("\n\n")
	}

	if f.Flags.IsSet(device.Write) {
		if f.Width == 1 {
			// Output an API that uses the `bool` data type  as the return type.
			builder.WriteString(boolSetter(f))
		} else {
			// Output an API that uses an integer data type of the necessary bit width as the return type.
			builder.WriteString(intSetter(f))
		}
		builder.WriteString("\n\n")
	}

	return output.WriteString(builder.String())
}

func boolGetter(f device.Field) string {
	dataType := "bool"
	if f.Constants != nil {
		dataType = typeName(f.Constants)
	}

	register := f.Register()
	registerWidth := max(8, device.NextPow2(register.Width))
	return fmt.Sprintf(`func (reg *%[1]s) Get%[2]s() %[5]s {
	return %[5]s(volatile.LoadUint%[3]d((*uint%[3]d)(reg))&(1<<%[4]d) != 0)
}`,
		typeName(register), formatSymbol(f.Identifier, true), registerWidth, f.Offset, dataType)
}

func boolSetter(f device.Field) string {
	dataType := "bool"
	if f.Constants != nil {
		dataType = typeName(f.Constants)
	}

	register := f.Register()
	registerWidth := max(8, device.NextPow2(register.Width))
	return fmt.Sprintf(`func (reg *%[1]s) Set%[2]s(enable %[5]s) {
	if enable {
		volatile.StoreUint%[3]d((*uint%[3]d)(reg), volatile.LoadUint%[3]d((*uint%[3]d)(reg))|(1<<%[4]d))
	} else {
		volatile.StoreUint%[3]d((*uint%[3]d)(reg), volatile.LoadUint%[3]d((*uint%[3]d)(reg))&^(1<<%[4]d))
	}
}`,
		typeName(register), formatSymbol(f.Identifier, true), registerWidth, f.Offset, dataType)
}

func intGetter(f device.Field) string {
	var returnType string
	if f.Constants != nil {
		returnType = typeName(f.Constants)
	} else {
		returnType = fmt.Sprintf("uint%d", max(8, device.NextPow2(f.Width)))
	}

	register := f.Register()
	registerWidth := max(8, device.NextPow2(register.Width))
	return fmt.Sprintf(`func (reg *%[1]s) Get%[2]s() %[3]s {
	return %[3]s(volatile.LoadUint%[4]d((*uint%[4]d)(reg))&%#[5]x) >> %[6]d
}`,
		typeName(register), formatSymbol(f.Identifier, true), returnType, registerWidth, device.Mask(f.Width, f.Offset), f.Offset)
}

func intSetter(f device.Field) string {
	var returnType string
	if f.Constants != nil {
		returnType = typeName(f.Constants)
	} else {
		returnType = fmt.Sprintf("uint%d", max(8, device.NextPow2(f.Width)))
	}

	register := f.Register()
	registerWidth := max(8, device.NextPow2(register.Width))
	return fmt.Sprintf(`func (reg *%[1]s) Set%[2]s(value %[3]s) {
	volatile.StoreUint%[4]d((*uint%[4]d)(reg), (volatile.LoadUint%[4]d((*uint%[4]d)(reg))&^%#[5]x)|(uint%[4]d(value)<<%[6]d))
}`,
		typeName(register), formatSymbol(f.Identifier, true), returnType, registerWidth, device.Mask(f.Width, f.Offset), f.Offset)
}

func varName(v any) string {
	switch v := v.(type) {
	case device.ConstantValue:
		f := v.ConstantGroup().Field()
		r := f.Register()
		return fmt.Sprintf("%s%s%s",
			formatSymbol(r.Identifier, true),
			formatSymbol(f.Identifier, true),
			formatSymbol(v.Identifier, true))
	case device.Peripheral:
		return formatSymbol(v.Identifier, true)
	case device.Register:
		return formatSymbol(v.Identifier, true)
	case device.RegisterGroup:
		return formatSymbol(v.Identifier, true)
	default:
		panic("unreachable")
	}
}

func typeName(v any) string {
	switch v := v.(type) {
	case device.ConstantGroup:
		return fmt.Sprintf("Constant%sType", formatSymbol(v.Identifier, true))
	case *device.ConstantGroup:
		return fmt.Sprintf("Constant%sType", formatSymbol(v.Identifier, true))
	case device.Peripheral:
		return fmt.Sprintf("Peripheral%sType", formatSymbol(v.Identifier, true))
	case device.Register:
		return fmt.Sprintf("Register%sType", formatSymbol(v.Identifier, true))
	case device.RegisterGroup:
		return fmt.Sprintf("RegisterGroup%sType", formatSymbol(v.Identifier, true))
	default:
		panic("unreachable")
	}
}
