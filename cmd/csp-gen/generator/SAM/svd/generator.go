package svd

import (
	"fmt"
	"go/format"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/exp/slices"

	"omibyte.io/sigo/cmd/csp-gen/generator"
	"omibyte.io/sigo/cmd/csp-gen/svd"
	"omibyte.io/sigo/cmd/csp-gen/types"
)

type samgen struct {
	device  *svd.DeviceElement
	methods map[string][]string
}

func NewGenerator(device *svd.DeviceElement) generator.Generator {
	return &samgen{
		device:  device,
		methods: map[string][]string{},
	}
}

func (s *samgen) Generate(out string) error {
	// Create the output directory for the chip
	outputDir := filepath.Join(out, "chip", strings.ToLower(s.device.Name))
	if err := os.MkdirAll(outputDir, 0750); err != nil {
		return err
	}

	// Generate the linker script first
	if err := s.generateLinkerScript(outputDir); err != nil {
		return err
	}

	var w strings.Builder

	// Write the preamble to the file
	if err := s.writePreamble(&w, "chip"); err != nil {
		return err
	}

	// Write required imports
	fmt.Fprintln(&w, "import (")
	fmt.Fprintln(&w, `_ "runtime/arm/cortexm"`)
	fmt.Fprintln(&w, `"unsafe"`)
	fmt.Fprintln(&w, `"volatile"`)
	fmt.Fprintln(&w, ")\n")

	// Generate all peripherals
	for _, periph := range s.device.Peripherals.Elements {
		if err := s.generatePeripheral(periph, &w); err != nil {
			return err
		}
	}

	fname := strings.ToLower(strings.ToLower(s.device.Name) + ".go")

	// Format the final output
	var buf []byte
	src := w.String()
	buf, err := format.Source([]byte(src))
	if err != nil {
		return fmt.Errorf("error formatting %s: %v", fname, err)
	}

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, "chip", fname))
	if err != nil {
		return err
	}
	f.Write(buf)
	f.Close()

	// Generate IRQ handlers
	if err := s.generateISR(outputDir); err != nil {
		return err
	}

	// Generate package init
	if err := s.generateInit(outputDir); err != nil {
		return err
	}

	return nil
}

func (s *samgen) generateInit(out string) (err error) {
	var w strings.Builder
	s.writePreamble(&w, filepath.Base(out))

	// Write required imports
	fmt.Fprintln(&w, "import (")
	fmt.Fprintln(&w, `_ "runtime/arm/cortexm"`)
	fmt.Fprintln(&w, ")")

	// Format the final output
	var buf []byte
	fname := filepath.Join(out, "init.go")
	src := w.String()
	if buf, err = format.Source([]byte(src)); err != nil {
		return fmt.Errorf("error formatting %s: %v", fname, err)
	}

	// Write the contents to the file
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	f.Write(buf)
	f.Close()

	return nil
}

func (s *samgen) generateLinkerScript(out string) (err error) {
	var w strings.Builder

	// Parse the name of the chip to extract the memory sizes
	regex := regexp.MustCompile(`(\d+)[A-Za-z]*$`)
	codes := regex.FindStringSubmatch(s.device.Name)

	ramSize := 0
	flashSize := 0

	switch codes[1] {
	case "14":
		flashSize = 16
		ramSize = 4
	case "15":
		flashSize = 32
		ramSize = 4
	case "16":
		flashSize = 64
		ramSize = 8
	case "17":
		flashSize = 128
		ramSize = 16
	case "18":
		flashSize = 256
		ramSize = 128
	case "19":
		flashSize = 512
		ramSize = 192
	case "20":
		flashSize = 1024
		ramSize = 256
	case "21":
		flashSize = 2048
		ramSize = 384
	default:
		return fmt.Errorf("cannot determine memories for %s", s.device.Name)
	}

	fmt.Fprintln(&w, "MEMORY")
	fmt.Fprintln(&w, "{")
	fmt.Fprintf(&w, "\tFLASH (rx) : ORIGIN = 0x00000000, LENGTH = %dK\n", flashSize)
	fmt.Fprintf(&w, "\tRAM (xrw)  : ORIGIN = 0x20000000, LENGTH = %dK\n", ramSize)
	fmt.Fprintln(&w, "}")
	fmt.Fprintln(&w, "__stack_size = 4K;")
	fmt.Fprintln(&w, "INCLUDE program.ld")

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, "target.ld"))
	if err != nil {
		return err
	}
	f.WriteString(w.String())
	f.Close()

	return nil
}

func (s *samgen) generateISR(out string) (err error) {
	var w strings.Builder

	// Write preamble
	w.WriteString(`.syntax unified

// This is the default handler for interrupts, if triggered but not defined.
.section .text.Default_Handler
.global  Default_Handler
.type    Default_Handler, %function
Default_Handler:
    wfe
    b    Default_Handler
.size Default_Handler, .-Default_Handler

// Avoid the need for repeated .weak and .set instructions.
.macro IRQ handler
    .weak  \handler
    .set   \handler, Default_Handler
.endm

// Must set the "a" flag on the section:
// https://svnweb.freebsd.org/base/stable/11/sys/arm/arm/locore-v4.S?r1=321049&r2=321048&pathrev=321049
// https://sourceware.org/binutils/docs/as/Section.html#ELF-Version
.section .isr_vector, "a", %progbits
.global  __isr_vector
__isr_vector:
    // Interrupt vector as defined by Cortex-M, starting with the stack top.
    // On reset, SP is initialized with *0x0 and PC is loaded with *0x4, loading
    // _stack_top and Reset_Handler.
    .long __stack
    .long Reset_Handler
	.long NMI_Handler
    .long HardFault_Handler
    .long MemoryManagement_Handler
    .long BusFault_Handler
    .long UsageFault_Handler
    .long 0
    .long 0
    .long 0
    .long 0
    .long SVC_Handler
    .long DebugMon_Handler
    .long 0
    .long PendSV_Handler
    .long SysTick_Handler

	IRQ NMI_Handler
    IRQ HardFault_Handler
    IRQ MemoryManagement_Handler
    IRQ BusFault_Handler
    IRQ UsageFault_Handler
    IRQ SVC_Handler
    IRQ DebugMon_Handler
    IRQ PendSV_Handler
    IRQ SysTick_Handler

`)

	// Collect all the interrupts
	interrupts := map[types.Integer]svd.InterruptElement{}
	irqMaxValue := types.Integer(0)
	for _, periph := range s.device.Peripherals.Elements {
		for _, irq := range periph.Interrupts {
			interrupts[irq.Value] = irq
			if irq.Value > irqMaxValue {
				irqMaxValue = irq.Value
			}
		}
	}

	// Fill the remainder of the vector table with the peripheral interrupt handlers
	for i := types.Integer(0); i < irqMaxValue; i++ {
		if irq, ok := interrupts[i]; ok {
			fmt.Fprintf(&w, "\t.long %s_Handler\n", irq.Name)
		} else {
			fmt.Fprintf(&w, "\t.long 0\n")
		}
	}

	fmt.Fprintln(&w, "\n")

	// Create the implementations next
	for _, irq := range interrupts {
		comment := ""
		if len(irq.Description) > 0 {
			comment = "/* " + irq.Description + " */"
		}
		fmt.Fprintf(&w, "\tIRQ %s_Handler %s\n", irq.Name, comment)
	}

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, "isr-vector.s"))
	if err != nil {
		return err
	}
	f.WriteString(w.String())
	f.Close()

	return nil
}

func (s *samgen) generatePeripheral(periph svd.PeripheralElement, w *strings.Builder) (err error) {
	// Don't create an implementation for derived peripherals
	if len(periph.DerivedFrom) > 0 {
		return nil
	}

	// Count the number of derived peripherals
	peripheralName := periph.Name
	var periphSeries []svd.PeripheralElement
	for _, p := range s.device.Peripherals.Elements {
		if p.DerivedFrom == periph.Name {
			periphSeries = append(periphSeries, p)
		}
	}

	if len(periphSeries) > 0 {
		// Prepend this peripheral to the slice of derived to complete the set
		periphSeries = append([]svd.PeripheralElement{periph}, periphSeries...)

		// Clean the name after determining the derived peripherals
		peripheralName = cleanIdentifier(periph.Name)
	} else {
		// Parse the name of the current peripheral to see if it is part of some series.
		re := regexp.MustCompile(`^([a-zA-Z]+)[0-9]+$`)
		matches := re.FindStringSubmatch(peripheralName)
		if len(matches) != 0 {
			// This peripheral is part of some truncated set for this chip within its product series
			periphSeries = append([]svd.PeripheralElement{periph}, periphSeries...)

			// Take the base name of this peripheral series
			peripheralName = matches[1]
		}
	}

	// Create the variable for this peripheral
	var registerImpls []string
	var clusterImpls []string
	if len(periphSeries) > 0 {
		strImpl, rimpls, cimpls, hasPointers := s.generatePeripheralStruct(periph)
		registerImpls = append(registerImpls, rimpls...)
		clusterImpls = append(clusterImpls, cimpls...)
		if hasPointers {
			fmt.Fprintf(w, "type Peripheral%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(w, "var (")
			fmt.Fprintf(w, "%s = [%d]Peripheral%s{\n", periph.Group, len(periphSeries), periph.Group)
			for _, p := range periphSeries {
				fmt.Fprintf(w, "{\n")
				for _, cluster := range periph.Registers.ClusterElements {
					clusterName := strings.ReplaceAll(cluster.Name, "[%s]", "")
					fmt.Fprintf(w, "%s: %s(unsafe.Pointer(uintptr(%#x))),\n", clusterName, periph.Group+clusterName, p.BaseAddress+cluster.AddressOffset)
				}
				fmt.Fprintf(w, "},\n")
			}
			fmt.Fprintln(w, "}")
			fmt.Fprintln(w, ")")
		} else {
			fmt.Fprintf(w, "type Peripheral%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(w, "var (")
			fmt.Fprintf(w, "%s = [%d]*Peripheral%s{\n", periph.Group, len(periphSeries), periph.Group)
			for _, p := range periphSeries {
				fmt.Fprintf(w, "(*Peripheral%s)(unsafe.Pointer(uintptr(%#x))),\n", periph.Group, p.BaseAddress)
			}
			fmt.Fprintln(w, "}")
			fmt.Fprintln(w, ")")
		}
	} else {
		strImpl, rimpls, cimpls, hasPointers := s.generatePeripheralStruct(periph)
		registerImpls = append(registerImpls, rimpls...)
		clusterImpls = append(clusterImpls, cimpls...)

		if hasPointers {
			fmt.Fprintf(w, "type Peripheral%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(w, "var (")
			fmt.Fprintf(w, "%s = Peripheral%s{\n", periph.Group, periph.Group)
			for _, cluster := range periph.Registers.ClusterElements {
				fmt.Fprintf(w, "%s: (%s)(unsafe.Pointer(uintptr(%#x))),\n", cluster.Name, peripheralName+cluster.Name, periph.BaseAddress+cluster.AddressOffset)
			}
			fmt.Fprintln(w, "}")
			fmt.Fprintln(w, ")")
		} else {
			fmt.Fprintln(w, "var (")
			fmt.Fprintln(w, "// ", peripheralName, periph.Description)
			fmt.Fprintf(w, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n", peripheralName, strImpl, periph.BaseAddress)
			fmt.Fprintln(w, ")")
		}
	}

	// Write each register implementation
	for _, impl := range clusterImpls {
		fmt.Fprintf(w, "%s\n", impl)
	}

	for _, impl := range registerImpls {
		fmt.Fprintf(w, "%s\n", impl)
	}

	/*for _, register := range periph.Registers.RegisterElements {
		_, impl := s.generateRegisterType(periph.Group, len(register.Alternative) > 0, register)
		fmt.Fprintln(w, impl)
	}*/

	return nil
}

func (s *samgen) writePreamble(w io.Writer, pkg string) error {
	// TODO: Write the license text

	// Write build tag
	fmt.Fprintf(w, "//go:build %s\n\n", strings.ToLower(s.device.Name))

	// Write the package
	fmt.Fprintln(w, "package ", pkg)

	return nil
}

func (s *samgen) generatePeripheralStruct(periph svd.PeripheralElement) (string, []string, []string, bool) {
	var buf strings.Builder
	var registerImpls []string
	var clusterImpls []string
	hasPointers := false

	// Format the peripheral name
	//peripheralName := cleanIdentifier(periph.Name)

	fmt.Fprintln(&buf, "struct {")
	offset := types.Integer(0)

	// Sort the registers
	slices.SortStableFunc(periph.Registers.RegisterElements, func(a, b svd.RegisterElement) int {
		return int(a.AddressOffset - b.AddressOffset)
	})

	// Collect registers and cluster into the same list, so they can be sorted by address offset
	var objs []svd.Addressable
	for _, cluster := range periph.Registers.ClusterElements {
		objs = append(objs, cluster)
	}

	for _, register := range periph.Registers.RegisterElements {
		objs = append(objs, register)
	}

	// Sort by address offset
	slices.SortStableFunc(objs, func(a, b svd.Addressable) int {
		return int(a.GetAddressOffset() - b.GetAddressOffset())
	})

	for _, obj := range objs {
		if obj.GetAddressOffset() > offset {
			// Insert padding bytes
			padding := obj.GetAddressOffset() - offset
			fmt.Fprintf(&buf, "_ [%d]byte\n", padding)
			offset += padding
		}

		switch obj := obj.(type) {
		case svd.ClusterElement:
			var clusterBuf strings.Builder
			clusterName := cleanIdentifier(obj.Name)

			count := types.Integer(1)
			if obj.Count > 0 {
				count = obj.Count
			}

			// Count all clusters with the same address offset as this one
			numShared := 0
			for _, other := range periph.Registers.ClusterElements {
				if other.GetAddressOffset() == obj.GetAddressOffset() {
					numShared++
				}
			}

			// Create a pointer to this cluster if more than 1 cluster that share the same  exist
			shouldBePointer := false
			if numShared > 1 {
				shouldBePointer = true
				hasPointers = true
			}

			// Create the cluster struct
			if shouldBePointer {
				fmt.Fprintf(&clusterBuf, "type %s%s *struct{\n", periph.Group, clusterName)
			} else {
				fmt.Fprintf(&clusterBuf, "type %s%s struct{\n", periph.Group, clusterName)
			}

			nestedOffset := types.Integer(0)
			for _, register := range obj.Registers {
				// Append to existing type for alternative registers
				if len(register.Alternative) > 0 {
					_, registerImpl := s.generateRegisterType(periph.Group+clusterName, true, register)
					registerImpls = append(registerImpls, registerImpl)
				} else {
					registerName := cleanIdentifier(register.Name)
					registerTypename, registerImpl := s.generateRegisterType(periph.Group+clusterName, false, register)

					if register.AddressOffset > nestedOffset {
						// insert padding bytes
						padding := register.AddressOffset - nestedOffset
						fmt.Fprintf(&clusterBuf, "_ [%d]byte\n", padding)
						nestedOffset += padding
					}

					if register.Count > 0 {
						fmt.Fprintf(&clusterBuf, "%s [%d]%s\n", registerName, register.Count, registerTypename)
						nestedOffset += (register.Size / s.device.AddressableWidth) * register.Count
					} else {
						fmt.Fprintf(&clusterBuf, "%s %s\n", registerName, registerTypename)
						nestedOffset += register.Size / s.device.AddressableWidth
					}
					registerImpls = append(registerImpls, registerImpl)
				}
			}
			// NOTE: The padding value is added below since it is not accounted for in nestedOffset
			offset += nestedOffset * count

			if count > 1 {
				// The padding should go in between elements in the resulting array
				padding := obj.Increment - nestedOffset
				fmt.Fprintf(&clusterBuf, "_ [%d]byte\n", obj.Increment-nestedOffset)
				offset += padding * count
			}

			fmt.Fprintln(&clusterBuf, "}\n")

			// Create the peripheral struct member
			if count > 1 {
				fmt.Fprintf(&buf, "%s [%d]%s%s\n", clusterName, obj.Count, periph.Group, clusterName)
			} else {
				fmt.Fprintf(&buf, "%s %s%s\n", clusterName, periph.Group, clusterName)
			}

			clusterImpls = append(clusterImpls, clusterBuf.String())
		case svd.RegisterElement:
			if obj.Alternative != "" {
				_, registerImpl := s.generateRegisterType(periph.Group, true, obj)
				registerImpls = append(registerImpls, registerImpl)
			} else {
				registerName := cleanIdentifier(obj.Name)
				count := types.Integer(1)
				if obj.Count > 0 {
					count = obj.Count
				}

				typename, registerImpl := s.generateRegisterType(periph.Group, false, obj)
				registerImpls = append(registerImpls, registerImpl)

				if count > 1 {
					fmt.Fprintf(&buf, "%s [%d]%s\n", registerName, obj.Count, typename)
				} else {
					fmt.Fprintf(&buf, "%s %s\n", registerName, typename)
				}

				offset += (obj.Size / s.device.AddressableWidth) * count
			}
		}
	}
	fmt.Fprint(&buf, "}")

	return buf.String(), registerImpls, clusterImpls, hasPointers
}

func (s *samgen) generateRegisterType(prefix string, omitType bool, register svd.RegisterElement) (string, string) {
	var buf strings.Builder
	typename := prefix + cleanIdentifier(register.Name)
	if len(register.Alternative) > 0 {
		typename = prefix + cleanIdentifier(register.Alternative)
	}
	receiver := strings.ToLower(typename[0:1])
	evMap := map[string]string{}

	storeFn := s.storeForSize(register.Size)
	loadFn := s.loadForSize(register.Size)
	registerDataType := s.typeForSize(register.Size)

	if !omitType {
		// Declare the type
		fmt.Fprintf(&buf, "type %s %s\n\n", typename, s.typeForSize(register.Size))

		// Create enumerated types
		for _, field := range register.Fields.Elements {
			fieldName := cleanIdentifier(field.Name)
			if len(field.EnumeratedValues.Elements) > 0 {
				evTypename, evImpl := s.generateEnumeratedValuesType(typename, field.EnumeratedValues)

				// Map the field name to the enumerated type's typename
				evMap[fieldName] = evTypename

				// Write the implementation
				fmt.Fprintln(&buf, evImpl)
			}
		}
	}

	// Create a setter/getter method for each field
	for _, field := range register.Fields.Elements {
		fieldName := cleanIdentifier(field.Name)
		fieldDataType := s.typeForSize(field.BitWidth)

		access := register.Access
		if len(field.Access) > 0 {
			// Override the access level of the register if explicitly set on the field
			access = field.Access
		} else if len(access) == 0 {
			// Use the default access level of the device if none set
			access = s.device.DefaultAccess
		}

		// Create methods
		if access == "read-only" || access == "read-write" || access == "read-writeOnce" {
			// Write the getter
			var returnType string
			methodName := fmt.Sprintf("Get%s", fieldName)

			// Check for collision (for alternative registers)
			if s.typeHasMethod(typename, methodName) {
				// Skip this
				// TODO: Should probably prefix this method instead just in case the field is different in some way
				continue
			} else {
				s.methods[typename] = append(s.methods[typename], methodName)
			}

			// Determine what the return type should be
			enumeratedType, hasEv := evMap[fieldName]
			if hasEv {
				returnType = enumeratedType
			} else {
				returnType = fieldDataType
				if field.BitWidth == 1 {
					returnType = "bool"
				}
			}

			fmt.Fprintf(&buf, "func (%s *%s) %s() %s {\n", receiver, typename, methodName, returnType)
			fmt.Fprintf(&buf, "v := %s((*%s)(%s))\n", loadFn, registerDataType, receiver)

			if hasEv {
				fmt.Fprintf(&buf, "return %s(v)>>%d", enumeratedType, field.BitOffset)
			} else {
				if field.BitWidth == 1 {
					fmt.Fprintf(&buf, "return %s(v&(1<<%d)) != 0\n", fieldDataType, field.BitOffset)
				} else {
					fmt.Fprintf(&buf, "return %s((v>>%d)&%s)\n", fieldDataType, field.BitOffset, allSet(field.BitWidth))
				}
			}
			fmt.Fprintf(&buf, "}\n\n")
		}

		if access == "write-only" || access == "read-write" || access == "writeOnce" {
			enumeratedType, hasEv := evMap[fieldName]
			paramType := typeForBitWidth(field.BitWidth)
			if hasEv {
				paramType = enumeratedType
			}

			methodName := fmt.Sprintf("Set%s", fieldName)
			// Check for collision (for alternative registers)
			if s.typeHasMethod(typename, methodName) {
				// Skip this
				// TODO: Should probably prefix this method instead just in case the field is different in some way
				continue
			} else {
				s.methods[typename] = append(s.methods[typename], methodName)
			}

			// Write the setter method
			fmt.Fprintf(&buf, "func (%s *%s) %s(value %s) {\n", receiver, typename, methodName, paramType)
			fmt.Fprintf(&buf, "v := %s((*%s)(%s))\n", loadFn, registerDataType, receiver)
			if field.BitWidth == 1 && !hasEv {
				fmt.Fprintln(&buf, "if value {")
				fmt.Fprintf(&buf, "%s((*%s)(%s), %s(v | (1 << %d)))\n", storeFn, registerDataType, receiver, registerDataType, field.BitOffset)
				fmt.Fprintln(&buf, "} else {")
				fmt.Fprintf(&buf, "%s((*%s)(%s), %s(%s(v) & (^%s(1 << %d))))\n", storeFn, registerDataType, receiver, registerDataType, typename, typename, field.BitOffset)
				fmt.Fprintln(&buf, "}")
			} else {
				fmt.Fprintf(&buf, "%s((*%s)(%s), %s((%s(v) & ^%s(%s << %d)) | %s(value) << %d))\n", storeFn, registerDataType, receiver, registerDataType, typename, typename, allSet(field.BitWidth), field.BitOffset, typename, field.BitOffset)
			}

			fmt.Fprintf(&buf, "}\n\n")
		}
	}
	return typename, buf.String()
}

func (s *samgen) generateEnumeratedValuesType(prefix string, ev svd.EnumeratedValuesElement) (string, string) {
	var buf strings.Builder
	typename := prefix + cleanIdentifier(ev.Name)

	// Declare the type
	fmt.Fprintf(&buf, "type %s uint32\n\n", typename)
	fmt.Fprintln(&buf, "const (")
	// Create the constant values
	for _, value := range ev.Elements {
		valueName := cleanIdentifier(value.Name)
		if len(value.Description) > 0 {
			fmt.Fprintf(&buf, "// %s %s\n", typename+valueName, value.Description)
		}
		fmt.Fprintf(&buf, "%s %s = %#x\n\n", typename+valueName, typename, value.Value)
	}
	fmt.Fprintln(&buf, ")")

	return typename, buf.String()
}

func (s *samgen) typeForSize(size types.Integer) string {
	switch {
	case size <= 8:
		return "uint8"
	case size <= 16:
		return "uint16"
	case size <= 32:
		return "uint32"
	default:
		return s.typeForSize(s.device.RegisterSize)
	}
}

func (s *samgen) storeForSize(size types.Integer) string {
	switch {
	case size <= 8:
		return "volatile.StoreUint8"
	case size <= 16:
		return "volatile.StoreUint16"
	case size <= 32:
		return "volatile.StoreUint32"
	default:
		return s.storeForSize(s.device.RegisterSize)
	}
}

func (s *samgen) loadForSize(size types.Integer) string {
	switch {
	case size <= 8:
		return "volatile.LoadUint8"
	case size <= 16:
		return "volatile.LoadUint16"
	case size <= 32:
		return "volatile.LoadUint32"
	default:
		return s.loadForSize(s.device.RegisterSize)
	}
}

func (s *samgen) typeHasMethod(typename, method string) bool {
	if methods, ok := s.methods[typename]; ok {
		for _, methodName := range methods {
			if methodName == method {
				return true
			}
		}
	}
	return false
}

func maxForSize(size types.Integer) string {
	switch size {
	case 8:
		return "0xFF"
	case 16:
		return "0xFFFF"
	case 32:
		return "0xFFFFFFFF"
	default:
		panic(fmt.Sprintf("unexpected size %d", size))
	}
}

func typeForBitWidth(width types.Integer) string {
	/*switch width {
	case 1:
		return "bool"
	default:

		return "uint32"
	}*/

	if width > 16 {
		return "uint32"
	} else if width > 8 {
		return "uint16"
	} else if width > 1 {
		return "uint8"
	} else {
		return "bool"
	}
}

func allSet(bits types.Integer) (result string) {
	for i := types.Integer(0); i < bits; i++ {
		result += "1"
	}
	result = "0b" + result
	return
}

func cleanIdentifier(ident string) string {
	re := regexp.MustCompile(`([a-zA-Z0-9]$|[a-zA-Z0-9][_a-zA-Z0-9]*[a-zA-Z0-9])`)
	cleanStr := re.FindStringSubmatch(ident)
	return cleanStr[0]
}
