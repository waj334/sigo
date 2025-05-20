package main

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"strings"

	"omibyte.io/sigo/targets/device/svd"
)

func translateSVD(ctx context.Context, fname string) error {
	// Open the input file.
	file, err := os.Open(fname)
	if err != nil {
		return err
	}

	// Read the file into memory.
	b, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Unmarshal in the SVD.
	var element svd.DeviceElement
	err = xml.Unmarshal(b, &element)
	if err != nil {
		return err
	}

	for _, peripheral := range element.Peripherals.Elements {
		_, err = translatePeripheral(ctx, os.Stdout, peripheral)
		if err != nil {
			return err
		}
	}

	return nil
}

func translatePeripheral(ctx context.Context, out io.Writer, peripheral svd.PeripheralElement) (int, error) {
	var builder strings.Builder

	defName := fmt.Sprintf("%sPeripheral", peripheral.Name)
	name := peripheral.Name
	description := ""
	if peripheral.Description != nil {
		description = strings.TrimSpace(*peripheral.Description)
	}

	fmt.Fprintf(&builder, "include \"base.td\"\n\n")

	if description != "" {
		fmt.Fprintf(&builder, "def %s : PeripheralType<\"%s\", \"%s\"> {\n", defName, name, description)
	} else {
		fmt.Fprintf(&builder, "def %s : PeripheralType<\"%s\"> {\n", defName, name)
	}

	fmt.Fprintf(&builder, "  let accessWidth = 32;\n")

	if len(peripheral.Registers.RegisterElements) > 0 {
		fmt.Fprintf(&builder, "  let registers = [\n")
		for _, reg := range peripheral.Registers.RegisterElements {
			regName := sanitizeName(reg.DisplayName, reg.Name)
			offset := reg.AddressOffset.Value()
			width := 32 // fallback default

			if reg.Size != 0 {
				width = int(reg.Size.Value())
			}

			regDesc := ""
			if len(reg.Description) > 0 {
				regDesc = strings.TrimSpace(reg.Description)
			}

			fmt.Fprintf(&builder, "    Register<\"%s\", %#x, %d, [\n", regName, offset, width)

			// Fields
			for _, field := range reg.Fields.Elements {
				fieldName := sanitizeName(field.Name, "")
				fieldDesc := ""
				if len(field.Description) > 0 {
					fieldDesc = strings.TrimSpace(field.Description)
				}

				offset, width := bitRangeToOffsetWidth(field.BitRange)

				access := accessMode(field.Access)

				fmt.Fprintf(&builder, "      Field<\"%s\", %d, %d, %s", fieldName, offset, width, access)

				if fieldDesc != "" {
					fmt.Fprintf(&builder, ", \"%s\"", fieldDesc)
				}

				namedEnums := []string{}
				for _, enum := range field.EnumeratedValues.Elements {
					enumName := sanitizeName(enum.Name, "")
					if enumName == "" {
						continue // skip unnamed enums
					}
					enumDesc := ""
					if len(enum.Description) > 0 {
						enumDesc = strings.TrimSpace(enum.Description)
					}
					val := enum.Value.Value()

					var enumBuilder strings.Builder
					fmt.Fprintf(&enumBuilder, "        Enum<\"%s\", %d", enumName, val)
					if enumDesc != "" {
						fmt.Fprintf(&enumBuilder, ", \"%s\"", enumDesc)
					}
					fmt.Fprintf(&enumBuilder, ">,\n")

					namedEnums = append(namedEnums, enumBuilder.String())
				}

				if len(namedEnums) > 0 {
					fmt.Fprintf(&builder, ", [\n")
					for _, enumStr := range namedEnums {
						builder.WriteString(enumStr)
					}
					fmt.Fprintf(&builder, "      ]")
				}

				fmt.Fprintf(&builder, ">,\n")
			}

			fmt.Fprintf(&builder, "    ], \"%s\">,\n", regDesc)
		}
		fmt.Fprintf(&builder, "  ];\n")
	}

	fmt.Fprintf(&builder, "}\n\n")
	return fmt.Fprint(out, builder.String())
}

func sanitizeName(display, fallback string) string {
	name := display
	if name == "" {
		name = fallback
	}
	name = strings.ReplaceAll(name, "%s", "")
	name = strings.ReplaceAll(name, "[%s]", "")
	name = strings.ReplaceAll(name, "[", "")
	name = strings.ReplaceAll(name, "]", "")
	name = strings.ReplaceAll(name, " ", "")
	name = strings.ReplaceAll(name, "-", "_")

	// Clean up trailing separators.
	for strings.HasSuffix(name, "_") {
		name = strings.TrimSuffix(name, "_")
	}

	return name
}

func accessMode(svdAccess string) string {
	switch strings.ToLower(svdAccess) {
	case "read-write":
		return "ReadWrite"
	case "read-only":
		return "Read"
	case "write-only":
		return "Write"
	default:
		return "ReadWrite"
	}
}

func bitRangeToOffsetWidth(br string) (offset, width int) {
	// Format: [high:low]
	var hi, lo int
	fmt.Sscanf(br, "[%d:%d]", &hi, &lo)
	return lo, hi - lo + 1
}
