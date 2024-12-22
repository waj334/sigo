package importer

import (
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"omibyte.io/sigo/cmd/csp-gen/generator"
	"omibyte.io/sigo/cmd/csp-gen/svd"
	"os"
)

func ImportSVD(fname string) (*generator.Device, error) {
	// Open the input file.
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}

	// Read the file into memory.
	b, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// Marshal in the SVD.
	var svdDevice svd.DeviceElement
	err = xml.Unmarshal(b, &svdDevice)
	if err != nil {
		return nil, err
	}

	device := new(generator.Device)

	// Populate peripherals.
	device.Peripherals = make([]generator.Register, len(svdDevice.Peripherals.Elements))
	for i, element := range svdDevice.Peripherals.Elements {
		peripheral := &device.Peripherals[i]

		// Handle clusters.
		for _, cluster := range element.Registers.ClusterElements {
			for _, register := range cluster.Registers {
				// Create the register struct value.
				reg := generator.Register{
					Identifier:  fmt.Sprintf("%s%s", cluster.Name, register.Name),
					Width:       uintptr(register.Size),
					Description: register.Description,
					Instances: []uintptr{
						uintptr(element.BaseAddress + register.AddressOffset),
					},
				}

				// Populate fields.
				reg.Fields = make([]generator.Field, len(register.Fields.Elements))
				for i, field := range register.Fields.Elements {
					f := &reg.Fields[i]
					if len(field.EnumeratedValues.Elements) > 0 {
						f.Constants.Identifier = field.EnumeratedValues.Name
						f.Constants.Values = make([]generator.ConstantValue, len(field.EnumeratedValues.Elements))
						for i, value := range field.EnumeratedValues.Elements {
							f.Constants.Values[i] = generator.ConstantValue{
								Identifier:  value.Name,
								Description: value.Description,
								Value:       uint64(value.Value),
							}
						}
					}

					f.Identifier = field.Name
					f.Description = field.Description
					f.Width = uintptr(field.BitWidth)
					f.Offset = uintptr(field.BitOffset)
					f.Flags = translateAccess(field.Access)

				}
			}
		}
		peripheral.Fields = make([]generator.Register, len(element.Registers.RegisterElements))
	}

	return device, nil
}

func translateAccess(v string) generator.AttributeFlag {
	switch v {
	case "read-only":
		return generator.Read
	case "write-only":
		return generator.Write
	case "read-write":
		return generator.ReadWrite
	}

	log.Panicf("unknown access value %s", v)
	return 0
}
