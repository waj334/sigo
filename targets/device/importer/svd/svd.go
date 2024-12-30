package svd

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strconv"

	"omibyte.io/sigo/targets/device"
	"omibyte.io/sigo/targets/device/importer"
	"omibyte.io/sigo/targets/device/svd"
)

type (
	clusterContextKey    struct{}
	deviceContextKey     struct{}
	fieldContextKey      struct{}
	peripheralContextKey struct{}
	registerContextKey   struct{}
)

func ImportSVD(ctx context.Context, config importer.Config) (d device.Device, err error) {
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

	// Unmarshal in the SVD.
	var element svd.DeviceElement
	err = xml.Unmarshal(b, &element)
	if err != nil {
		return device.Device{}, err
	}

	ctx = context.WithValue(context.Background(), deviceContextKey{}, element)

	if element.DefaultAccess != nil {
		d.Flags = translateAccess(*element.DefaultAccess)
	} else {
		d.Flags.Set(device.Read, device.Write)
	}

	// Populate peripherals.
	d.Peripherals = make([]device.Peripheral, len(element.Peripherals.Elements))
	for i, peripheral := range element.Peripherals.Elements {
		d.Peripherals[i] = translatePeripheral(ctx, d, peripheral)
	}

	// Attempt to merge peripheral instances by best effort.
	nameRegex := regexp.MustCompile("^([a-zA-Z0-9]+)([0-9]+)$")
	mergeMap := map[string][]int{}
	var unmatched []int
	for i, p := range d.Peripherals {
		// Match the name using the regex.
		matcheGroups := nameRegex.FindStringSubmatch(p.Identifier)
		if matcheGroups != nil {
			group := mergeMap[matcheGroups[1]]
			instance, _ := strconv.Atoi(matcheGroups[2])
			if len(group) < instance+1 {
				newGroup := make([]int, instance+1)
				for i := range newGroup {
					newGroup[i] = -1
				}

				for i, g := range group {
					newGroup[i] = g
				}
				group = newGroup
			}
			group[instance] = i
			mergeMap[matcheGroups[1]] = group
		} else {
			unmatched = append(unmatched, i)
		}
	}

	var newPeripherals []device.Peripheral
	for _, i := range unmatched {
		newPeripherals = append(newPeripherals, d.Peripherals[i])
	}

	for group, indices := range mergeMap {
		// Remove negative indices.
		indices = slices.DeleteFunc(indices, func(i int) bool {
			return i == -1
		})

		if len(indices) > 0 {
			// Make a copy of the first instance.
			base := d.Peripherals[indices[0]]
			base.Instances = []uintptr{*base.BaseAddress}
			base.Identifier = group

			for _, i := range indices[1:] {
				p := d.Peripherals[i]

				// Add the remaining as instances.
				base.Instances = append(base.Instances, *p.BaseAddress)

				// Merge interrupts.
				base.Interrupts = append(base.Interrupts, p.Interrupts...)
			}

			slices.Sort(base.Instances)

			// Add to the new peripherals list.
			newPeripherals = append(newPeripherals, base)
		}
	}

	// Sort the peripherals by base address.
	slices.SortFunc(newPeripherals, func(a device.Peripheral, b device.Peripheral) int {
		return int(*a.BaseAddress - *b.BaseAddress)
	})

	// Unset the base address if there are instances.
	for _, p := range newPeripherals {
		if len(p.Instances) > 0 {
			p.BaseAddress = nil
		}
	}

	// Replace the peripherals list.
	d.Peripherals = newPeripherals

	return d, nil
}

func translatePeripheral(ctx context.Context, d device.Device, element svd.PeripheralElement) (p device.Peripheral) {
	ctx = context.WithValue(ctx, peripheralContextKey{}, element)

	if element.DerivedFrom != nil {
		deviceElement := ctx.Value(deviceContextKey{}).(svd.DeviceElement)
		index, found := deviceElement.Peripherals.Find(*element.DerivedFrom)
		if found {
			p = translatePeripheral(ctx, d, deviceElement.Peripherals.Elements[index])
		} else {
			panic("base peripheral not found")
		}
	}

	// Set immutable values.
	set(&p.Identifier, element.Name)
	set(&p.Description, element.Description)
	set(&p.BaseAddress, element.BaseAddress)

	p.Description = device.CleanDescription(p.Description)

	if element.Access != nil {
		p.Flags = translateAccess(*element.Access)
	} else {
		p.Flags = d.Flags
	}

	// Translate registers.
	if element.Registers != nil {
		registers := make([]device.Register, len(element.Registers.RegisterElements))
		for i, register := range element.Registers.RegisterElements {
			registers[i] = translateRegister(ctx, p, register)
		}

		// TODO: Correct this!
		//p.Registers = append(p.Registers, registers...)

		// Translate register clusters.
		for _, cluster := range element.Registers.ClusterElements {
			ctx := context.WithValue(ctx, clusterContextKey{}, cluster)
			registers := make([]device.Register, len(cluster.Registers))
			for i, register := range cluster.Registers {
				registers[i] = translateRegister(ctx, p, register)
			}
		}
	}

	if element.Interrupts != nil {
		p.Interrupts = make([]device.Interrupt, len(*element.Interrupts))
		for i, interrupt := range *element.Interrupts {
			p.Interrupts[i] = translateInterrupt(ctx, p, interrupt)
		}
	}
	return
}

func translateInterrupt(ctx context.Context, peripheral device.Peripheral, element svd.InterruptElement) (i device.Interrupt) {
	i.Identifier = element.Name
	i.Description = device.CleanDescription(element.Description)
	i.Number = int(element.Value)
	return
}

func translateRegister(ctx context.Context, peripheral device.Peripheral, element svd.RegisterElement) (r device.Register) {
	ctx = context.WithValue(ctx, registerContextKey{}, element)

	// Set immutable values.
	r.Identifier = element.Name
	r.Description = device.CleanDescription(element.Description)
	r.Width = uintptr(element.Size)
	r.Flags = translateAccess(element.Access)
	if r.Flags.IsUnset() {
		r.Flags = peripheral.Flags
	}

	// Translate fields.
	r.Fields = make([]device.Field, len(element.Fields.Elements))
	for i, field := range element.Fields.Elements {
		r.Fields[i] = translateField(ctx, r, field)
	}

	return
}

func translateField(ctx context.Context, register device.Register, element svd.FieldElement) (f device.Field) {
	ctx = context.WithValue(ctx, fieldContextKey{}, element)

	// Set immutable values.
	f.Identifier = element.Name
	f.Description = device.CleanDescription(element.Description)
	f.Width = uintptr(element.BitWidth)
	f.Offset = uintptr(element.BitOffset)
	f.Flags = translateAccess(element.Access)
	if f.Flags.IsUnset() {
		f.Flags = register.Flags
	}

	// Translate constant enumerations.
	if len(element.EnumeratedValues.Elements) > 0 {
		f.Constants = new(device.ConstantGroup)
		*f.Constants = translateEnumerations(ctx, f, element.EnumeratedValues)
	}

	return
}

func translateEnumerations(ctx context.Context, field device.Field, element svd.EnumeratedValuesElement) (c device.ConstantGroup) {
	c.Identifier = element.Name
	c.Values = make([]device.ConstantValue, len(element.Elements))
	for i, enum := range element.Elements {
		c.Values[i] = device.ConstantValue{
			Identifier:  enum.Name,
			Description: device.CleanDescription(enum.Description),
			Value:       uint64(enum.Value),
		}
	}
	return
}

func translateAccess(v string) device.AttributeFlags {
	switch v {
	case "read-only":
		return device.AttributeFlags{device.Read}
	case "write-only":
		return device.AttributeFlags{device.Write}
	case "read-write":
		return device.AttributeFlags{device.Read, device.Write}
	case "":
		return device.AttributeFlags{}
	}
	panic(fmt.Errorf("unknown access level %s", v))
}

func set[T any, VT any](addr *T, value *VT) {
	if value == nil {
		return
	}

	vVal := reflect.ValueOf(value).Elem() // reflect value of the pointed-to 'VT'
	aVal := reflect.ValueOf(addr).Elem()  // reflect value of the pointed-to 'T'

	// Check if vVal's type is convertible to aVal's type
	if !vVal.Type().ConvertibleTo(aVal.Type()) {
		panic(fmt.Sprintf(
			"cannot convert type %s to %s",
			vVal.Type().String(),
			aVal.Type().String()))
	}

	// Perform the conversion via reflection, then set
	converted := vVal.Convert(aVal.Type())
	aVal.Set(converted)
}
