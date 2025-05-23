package main

import (
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"pkg.si-go.dev/sigo/targets/device/svd"
)

func translateSVD(ctx context.Context, inputFilename string) error {
	// Open the input file.
	file, err := os.Open(inputFilename)
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

	// Map instances to their respective base peripheral.
	instanceMap := map[int][]svd.PeripheralElement{}
	for i, peripheral := range element.Peripherals.Elements {
		if peripheral.DerivedFrom != nil {
			derivedFrom := *peripheral.DerivedFrom
			index, ok := element.Peripherals.Find(derivedFrom)
			if ok {
				instanceMap[index] = append(instanceMap[index], peripheral)
			}
		} else {
			instanceMap[i] = append(instanceMap[i], peripheral)
		}
	}

	var includes []string
	for baseIndex, instances := range instanceMap {
		// Look up the base peripheral that the instances will be created from.
		basePeripheral := element.Peripherals.Elements[baseIndex]

		// Format the base filename.
		peripheralName := sanitizeName(basePeripheral.Name, basePeripheral.Name)
		peripheralName = strings.ToLower(peripheralName)

		// Format the path to the peripheral TableGen file.
		filename := filepath.Join(outputDirectory, "peripherals", peripheralName+".td")

		// Add this file to the list of includes.
		includes = append(includes, filename)

		// Create the directory.
		err = os.MkdirAll(filepath.Dir(filename), os.ModePerm)
		if err != nil {
			return err
		}

		// Create the peripheral TableGen file that will be written to.
		file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, os.ModePerm)
		if err != nil {
			return err
		}

		// Write the contents of the peripheral TableGen file.
		_, err = generatePeripheral(ctx, file, basePeripheral, instances)
		if err != nil {
			return err
		}
	}

	// Format the base filename.
	deviceName := sanitizeName(element.Name, element.Name)
	deviceName = strings.ToLower(deviceName)

	// Format the path to the series TableGen file.
	filename := filepath.Join(outputDirectory, deviceName+".td")

	// Create the directory.
	err = os.MkdirAll(filepath.Dir(filename), os.ModePerm)
	if err != nil {
		return err
	}

	// Create the peripheral TableGen file that will be written to.
	file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, os.ModePerm)
	if err != nil {
		return err
	}

	// Write the contents of the peripheral TableGen file.
	_, err = generateSeries(ctx, file, element, includes)
	if err != nil {
		return err
	}

	return nil
}

func generatePeripheral(ctx context.Context, out io.Writer, peripheral svd.PeripheralElement, instances []svd.PeripheralElement) (int, error) {
	var builder strings.Builder

	defName := fmt.Sprintf("%sPeripheral", peripheral.Name)
	name := sanitizeName(peripheral.Name, peripheral.Name)
	description := ""
	if peripheral.Description != nil {
		description = sanitizeDescription(*peripheral.Description)
	}

	if len(peripheral.Registers.RegisterElements) == 0 {
		return 0, nil
	}

	fmt.Fprintf(&builder, "#ifndef _PERIPHERALS_%s_TD\n", strings.ToUpper(name))
	fmt.Fprintf(&builder, "#define _PERIPHERALS_%s_TD\n\n", strings.ToUpper(name))
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

			count := 1
			if reg.Count > 1 {
				count = int(reg.Count.Value())
			}

			regDesc := ""
			if len(reg.Description) > 0 {
				regDesc = sanitizeDescription(reg.Description)
			}

			if count > 1 {
				fmt.Fprintf(&builder, "    RepeatingRegister<\"%s\", %d, %#x, %d, [\n", regName, count, offset, width)
			} else {
				fmt.Fprintf(&builder, "    Register<\"%s\", %#x, %d, [\n", regName, offset, width)
			}

			// Fields
			for _, field := range reg.Fields.Elements {
				fieldName := sanitizeName(field.Name, "")
				fieldDesc := ""
				if len(field.Description) > 0 {
					fieldDesc = sanitizeDescription(field.Description)
				}

				var offset int
				var width int
				if len(field.BitRange) > 0 {
					offset, width = bitRangeToOffsetWidth(field.BitRange)
				} else {
					offset = int(field.BitOffset.Value())
					width = int(field.BitWidth.Value())
				}

				access := accessMode(field.Access)

				// TODO: Implement repeating register fields.
				// TODO: Repeating register fields must respect the register's access width.

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
						enumDesc = sanitizeDescription(enum.Description)
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

	if len(instances) > 0 {
		instanceClassName := fmt.Sprintf("%sInstance", name)
		fmt.Fprintf(&builder, "class %s<string Name, int Base> : PeripheralInstance<Name, Base, %s>;\n", instanceClassName, defName)
		for _, instance := range instances {
			instanceName := sanitizeName(instance.Name, instance.Name)
			fmt.Fprintf(&builder, "def %s : %s<\"%s\", %#x>;\n", instanceName, instanceClassName, instanceName, instance.BaseAddress.Value())
		}
		fmt.Fprintf(&builder, "\n")
	}

	fmt.Fprintf(&builder, "#endif // _PERIPHERALS_%s_TD\n", strings.ToUpper(name))

	return fmt.Fprint(out, builder.String())
}

func generateSeries(ctx context.Context, out io.Writer, device svd.DeviceElement, includes []string) (int, error) {
	var builder strings.Builder

	name := sanitizeName(device.Name, device.Name)

	fmt.Fprintf(&builder, "#ifndef _%s_TD\n", strings.ToUpper(name))
	fmt.Fprintf(&builder, "#define _%s_TD\n\n", strings.ToUpper(name))
	fmt.Fprintf(&builder, "include \"base.td\"\n")

	var arch string
	switch strings.ToUpper(device.CPU.Name) {
	case "CM0":
		arch = "CortexM0"
	case "CM0PLUS":
		arch = "CortexM0Plus"
	case "CM0+":
		arch = "CortexM0Plus"
	case "CM1":
		arch = "CortexM1"
	case "CM3":
		arch = "CortexM3"
	case "CM4":
		arch = "CortexM4"
	case "CM7":
		arch = "CortexM7"
		fmt.Fprintf(&builder, "include \"arm/cortexm/interrupts.td\"\n")
		fmt.Fprintf(&builder, "include \"arm/cortexm/registers.td\"\n")
		fmt.Fprintf(&builder, "include \"arm/cortexm/variant.td\"\n")
		fmt.Fprintf(&builder, "include \"arm/family.td\"\n")
	case "CM23":
		arch = "CortexM23"
	case "CM33":
		arch = "CortexM33"
	case "CM35P":
		arch = "CortexM35P"
	case "CM52":
		arch = "CortexM52"
	case "CM55":
		arch = "CortexM55"
	case "CM85":
		arch = "CortexM85"
	default:
		return 0, errors.New("unknown architecture " + device.CPU.Name)
	}

	// Generate includes section.
	fmt.Fprintf(&builder, "\n")
	for _, include := range includes {
		include, err := filepath.Rel(outputDirectory, include)
		if err != nil {
			return 0, err
		}

		fmt.Fprintf(&builder, "include \"%s\"\n", include)
	}

	fmt.Fprintf(&builder, "\n")
	fmt.Fprintf(&builder, "def %s : Series<\"%s\", %s> {\n", name, name, arch)
	fmt.Fprintf(&builder, "  let variants = [];\n")
	fmt.Fprintf(&builder, "}\n")

	fmt.Fprintf(&builder, "#endif // _%s_TD\n", strings.ToUpper(name))
	return fmt.Fprint(out, builder.String())
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
