package atdf

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strings"

	"omibyte.io/sigo/targets/device"
	"omibyte.io/sigo/targets/device/atdf"
	"omibyte.io/sigo/targets/device/importer"
)

type (
	moduleContextKey     struct{}
	moduleModeContextKey struct{}
)

func ImportATDF(ctx context.Context, config importer.Config) (d device.Device, err error) {
	// Open the input file.
	file, err := os.Open(config.BaseFilename)
	if err != nil {
		return device.Device{}, err
	}

	// Read the file into memory.
	b, err := io.ReadAll(file)
	if err != nil {
		return device.Device{}, err
	}

	// Unmarshal in the ATDF.
	var root atdf.ATDF
	err = xml.Unmarshal(b, &root)
	if err != nil {
		return device.Device{}, err
	}

	d.Flags.Set(device.Read, device.Write)

	instances := map[string][]uintptr{}
	d.Variants = make([]device.Variant, len(root.Devices.Elements))
	for i, deviceElement := range root.Devices.Elements {
		variant := &d.Variants[i]
		variant.Identifier = strings.ToLower(deviceElement.Name)

		variant.Interrupts = make([]device.Interrupt, len(deviceElement.Interrupts.Elements))
		for i, irq := range deviceElement.Interrupts.Elements {
			variant.Interrupts[i] = device.Interrupt{
				Identifier:  cleanIdentifier(irq.Name),
				Description: device.CleanDescription(irq.Caption),
				Number:      int(irq.Index),
			}
		}

		startAddresses := map[string]uintptr{}
		if len(deviceElement.AddressSpaces.Elements) > 0 {
			addressSpace := deviceElement.AddressSpaces.Elements[0]
			variant.Memories = make([]device.Memory, len(addressSpace.MemorySegments))
			startAddresses[addressSpace.Name] = uintptr(addressSpace.Start)
			for i, memory := range addressSpace.MemorySegments {
				variant.Memories[i] = device.Memory{
					Identifier: cleanIdentifier(memory.Name),
					Type:       translateMemoryType(memory.Type),
					Start:      uintptr(memory.Start),
					Size:       uintptr(memory.Size),
					Flags:      translateRW(memory.RW),
				}
			}
		}

		for _, peripheral := range deviceElement.Peripherals.Modules {
			for _, instance := range peripheral.Instances {
				if len(instance.RegisterGroups) > 0 {
					l := instances[peripheral.Id]
					registerGroup := instance.RegisterGroups[0]
					start := startAddresses[registerGroup.AddressSpace]
					l = append(l, start+uintptr(registerGroup.Offset))

					if len(peripheral.Id) > 0 {
						instances[peripheral.Id] = l
					} else {
						instances[peripheral.Name] = l
					}
				}
			}
		}
	}

	if config.OnlyVariants {
		return
	}

	for _, module := range root.Modules.Elements {
		var instanceAddrs []uintptr

		if len(module.Id) > 0 {
			instanceAddrs = instances[module.Id]
		} else {
			instanceAddrs = instances[module.Name]
		}

		slices.Sort(instanceAddrs)

		groups := map[string]atdf.ModeElement{}
		for _, registerGroup := range module.RegisterGroups {
			if len(registerGroup.Modes) > 0 {
				for _, mode := range registerGroup.Modes {
					groups[mode.Name] = mode
				}
			}
		}

		if len(groups) > 0 {
			for _, mode := range groups {
				ctx := context.WithValue(ctx, moduleModeContextKey{}, mode)
				p := translatePeripheral(ctx, module)

				if len(p.RegisterGroups) > 0 {
					p.Instances = instanceAddrs
					d.Peripherals = append(d.Peripherals, p)
				}
			}
		} else {
			p := translatePeripheral(ctx, module)
			if len(p.RegisterGroups) > 0 {
				p.Instances = instanceAddrs
				d.Peripherals = append(d.Peripherals, p)
			}
		}
	}

	return
}

func translatePeripheral(ctx context.Context, element atdf.ModuleElement) (p device.Peripheral) {
	ctx = context.WithValue(ctx, moduleContextKey{}, element)

	var mode atdf.ModeElement
	hasMode := false
	if v := ctx.Value(moduleModeContextKey{}); v != nil {
		mode = v.(atdf.ModeElement)
		hasMode = true
	}

	if hasMode {
		p.Identifier = cleanIdentifier(mode.Name)
		p.Group = cleanIdentifier(element.Name)
		p.Description = device.CleanDescription(fmt.Sprintf("%s - %s", element.Caption, mode.Caption))
	} else {
		p.Identifier = cleanIdentifier(element.Name)
		p.Description = device.CleanDescription(element.Caption)
	}

	p.Flags.Set(device.Read, device.Write)

	for _, group := range element.RegisterGroups {
		registerGroup := translateRegisterGroup(ctx, group)
		p.RegisterGroups = append(p.RegisterGroups, registerGroup)
	}

	return
}

func translateRegisterGroup(ctx context.Context, element atdf.ModuleRegisterGroupElement) (r device.RegisterGroup) {
	var mode atdf.ModeElement
	hasMode := false
	if v := ctx.Value(moduleModeContextKey{}); v != nil {
		mode = v.(atdf.ModeElement)
		hasMode = true
	}

	r = device.RegisterGroup{
		Identifier:  element.Name,
		Reference:   element.NameInModule,
		Description: device.CleanDescription(element.Caption),
		Count:       max(1, int(element.Count)),
		Size:        uintptr(element.Size),
		Offset:      uintptr(element.Offset()),
	}

	for _, subregisterGroup := range element.Groups {
		subgroup := translateRegisterGroup(ctx, subregisterGroup)
		r.Groups = append(r.Groups, subgroup)
	}

	for _, register := range element.Registers {
		if hasMode && len(register.Mode) > 0 {
			modes := strings.Split(register.Mode, ",")
			if !slices.Contains(modes, mode.Name) {
				// Skip this register.
				continue
			}
		}

		reg := translateRegister(ctx, register)
		r.Registers = append(r.Registers, reg)
	}

	return
}

func translateRegister(ctx context.Context, element atdf.RegisterElement) (r device.Register) {
	moduleElement := ctx.Value(moduleContextKey{}).(atdf.ModuleElement)

	r = device.Register{
		Identifier:  cleanIdentifier(element.Name),
		Width:       uintptr(element.Size) * 8,
		Offset:      uintptr(element.Offset()),
		Count:       max(1, int(element.Count)),
		Description: device.CleanDescription(element.Caption),
		Flags:       translateRW(element.RW),
	}

	r.Fields = make([]device.Field, len(element.BitFields))
	for i, field := range element.BitFields {
		numBits := uintptr(0)
		offset := uintptr(0)
		if field.Mask != 0 {
			for n := range 64 {
				bit := (field.Mask >> n) & 1
				if bit == 0 {
					if numBits == 0 {
						offset++
					} else {
						// Stop examining the bits.
						break
					}
				} else {
					numBits++
				}
			}
		}

		r.Fields[i] = device.Field{
			Identifier:  cleanIdentifier(field.Name),
			Width:       numBits,
			Offset:      offset,
			Flags:       r.Flags,
			Description: device.CleanDescription(field.Caption),
		}

		if field.Values != nil {
			if valueGroup := moduleElement.FindValueGroup(*field.Values); valueGroup != nil {
				constantGroup := device.ConstantGroup{
					Identifier: cleanIdentifier(valueGroup.Name),
				}

				constantGroup.Values = make([]device.ConstantValue, len(valueGroup.Elements))
				for i, value := range valueGroup.Elements {
					constantGroup.Values[i] = device.ConstantValue{
						Identifier:  cleanIdentifier(value.Name),
						Description: device.CleanDescription(value.Caption),
						Value:       uint64(value.Value),
					}
				}
				r.Fields[i].Constants = &constantGroup
			}
		}
	}

	return
}

func translateRW(input string) device.AttributeFlags {
	switch input {
	case "R":
		return device.AttributeFlags{device.Read}
	case "W":
		return device.AttributeFlags{device.Write}
	case "RW":
		return device.AttributeFlags{device.Read, device.Write}
	default:
		return device.AttributeFlags{}
	}
}

func cleanIdentifier(input string) string {
	{
		regex := regexp.MustCompile("^_*([a-zA-Z0-9]+_*[a-zA-Z0-9]+)_+$")
		matches := regex.FindStringSubmatch(input)
		if matches != nil {
			input = matches[1]
		}
	}

	{
		// Replace all consecutive underscores.
		regex := regexp.MustCompile("(_+)")
		input = regex.ReplaceAllLiteralString(input, "_")
	}
	return input
}

func translateMemoryType(input string) device.MemoryType {
	switch strings.ToLower(input) {
	case "flash":
		return device.MemoryFlash
	case "fuses":
		return device.MemoryFuses
	case "user_page":
		return device.MemoryUser
	case "io":
		return device.MemoryIO
	case "ram":
		return device.MemoryRAM
	default:
		return device.MemoryUnknown
	}
}
