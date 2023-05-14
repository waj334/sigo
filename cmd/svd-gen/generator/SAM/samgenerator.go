package SAM

import (
	"fmt"
	"go/format"
	"golang.org/x/exp/slices"
	"io"
	"omibyte.io/sigo/cmd/svd-gen/generator"
	"omibyte.io/sigo/cmd/svd-gen/svd"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type samgen struct {
	device svd.DeviceElement
}

func NewGenerator(device svd.DeviceElement) generator.Generator {
	return &samgen{
		device: device,
	}
}

func (s *samgen) Generate(out string) error {
	// Create the output directory for the chip
	outputDir := filepath.Join(out, strings.ToLower(s.device.Name))
	if err := os.MkdirAll(outputDir, 0750); err != nil {
		return err
	}

	// Generate the linker script first
	if err := s.generateLinkerScript(outputDir); err != nil {
		return err
	}

	// Generate all peripherals
	for _, periph := range s.device.Peripherals.Elements {
		if err := s.generatePeripheral(periph, outputDir); err != nil {
			return err
		}
	}

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

	s.writePreamble(&w)

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
`)

	// Collect all the interrupts
	var interrupts []svd.InterruptElement
	for _, periph := range s.device.Peripherals.Elements {
		for _, irq := range periph.Interrupts {
			interrupts = append(interrupts, irq)
		}
	}

	// Sort the interrupts by value
	slices.SortStableFunc(interrupts, func(a, b svd.InterruptElement) bool {
		return a.Value < b.Value
	})

	// Fill the remainder of the vector table with the peripheral interrupt handlers
	for _, irq := range interrupts {
		comment := ""
		if len(irq.Description) > 0 {
			comment = "/* " + irq.Description + " */"
		}
		fmt.Fprintf(&w, "\tIRQ %s_Handler %s\n", irq.Name, comment)
	}

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, "__isr_vector.s"))
	if err != nil {
		return err
	}
	f.WriteString(w.String())
	f.Close()

	return nil
}

func (s *samgen) generatePeripheral(periph svd.PeripheralElement, out string) (err error) {
	var w strings.Builder

	// Don't create an implementation for derived peripherals
	if len(periph.DerivedFrom) > 0 {
		return nil
	}

	// Write the preamble to the file
	if err = s.writePreamble(&w); err != nil {
		return err
	}

	// Write package imports
	fmt.Fprintln(&w, "import (")
	fmt.Fprintln(&w, `"unsafe"`)
	fmt.Fprintln(&w, ")\n")

	// Count the number of derived peripherals
	peripheralName := periph.Name
	var derived []svd.PeripheralElement
	for _, p := range s.device.Peripherals.Elements {
		if p.DerivedFrom == periph.Name {
			derived = append(derived, p)
		}
	}

	// Create the variable for this peripheral
	var registerImpls []string
	var clusterImpls []string
	if len(derived) > 0 {
		// Prepend this peripheral to the slice of derived to complete the set
		derived = append([]svd.PeripheralElement{periph}, derived...)

		peripheralName = periph.Name[:len(peripheralName)-1]
		strImpl, rimpls, cimpls, hasPointers := s.generatePeripheralStruct(periph)
		registerImpls = append(registerImpls, rimpls...)
		clusterImpls = append(clusterImpls, cimpls...)
		if hasPointers {
			fmt.Fprintf(&w, "type S%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(&w, "var (")
			fmt.Fprintf(&w, "%s = [%d]S%s{\n", periph.Group, len(derived), periph.Group)
			for _, p := range derived {
				fmt.Fprintf(&w, "S%s{\n", periph.Group)
				for _, cluster := range periph.Registers.ClusterElements {
					clusterName := strings.ReplaceAll(cluster.Name, "[%s]", "")
					fmt.Fprintf(&w, "%s: S%s(unsafe.Pointer(uintptr(%#x))),\n", clusterName, periph.Group+clusterName, p.BaseAddress+cluster.AddressOffset)
				}
				fmt.Fprintf(&w, "},\n")
			}
			fmt.Fprintln(&w, "}")
			fmt.Fprintln(&w, ")")
		} else {
			fmt.Fprintf(&w, "type S%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(&w, "var (")
			fmt.Fprintf(&w, "%s = [%d]*S%s{\n", periph.Group, len(derived), periph.Group)
			for _, p := range derived {
				fmt.Fprintf(&w, "(*S%s)(unsafe.Pointer(uintptr(%#x))),\n", peripheralName, p.BaseAddress)
			}
			fmt.Fprintln(&w, "}")
			fmt.Fprintln(&w, ")")
		}
	} else {
		strImpl, rimpls, cimpls, hasPointers := s.generatePeripheralStruct(periph)
		registerImpls = append(registerImpls, rimpls...)
		clusterImpls = append(clusterImpls, cimpls...)

		if hasPointers {
			fmt.Fprintf(&w, "type S%s %s\n", periph.Group, strImpl)
			fmt.Fprintln(&w, "var (")
			fmt.Fprintf(&w, "%s = S%s{\n", periph.Group, periph.Group)
			for _, cluster := range periph.Registers.ClusterElements {
				fmt.Fprintf(&w, "%s: (S%s)(unsafe.Pointer(uintptr(%#x))),\n", cluster.Name, peripheralName+cluster.Name, periph.BaseAddress+cluster.AddressOffset)
			}
			fmt.Fprintln(&w, "}")
			fmt.Fprintln(&w, ")")
		} else {
			fmt.Fprintln(&w, "var (")
			fmt.Fprintf(&w, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n", peripheralName, strImpl, periph.BaseAddress)
			fmt.Fprintln(&w, ")")
		}
	}

	// Write each register implementation
	for _, impl := range clusterImpls {
		fmt.Fprintf(&w, "%s\n", impl)
	}

	for _, impl := range registerImpls {
		fmt.Fprintf(&w, "%s\n", impl)
	}

	for _, register := range periph.Registers.RegisterElements {
		_, impl := s.generateRegisterType(peripheralName, register)
		fmt.Fprintln(&w, impl)
	}

	fname := strings.ToLower(peripheralName + ".go")

	// Format the final output
	var buf []byte
	src := w.String()
	if buf, err = format.Source([]byte(src)); err != nil {
		return fmt.Errorf("error formatting %s: %v", fname, err)
	}

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, fname))
	if err != nil {
		return err
	}
	f.Write(buf)
	f.Close()

	return nil
}

func (s *samgen) writePreamble(w io.Writer) error {
	// TODO: Write the license text

	// Write the package
	if _, err := fmt.Fprintf(w, "package %s\n\n", strings.ToLower(s.device.Name)); err != nil {
		return err
	}

	return nil
}

func (s *samgen) generatePeripheralStruct(periph svd.PeripheralElement) (string, []string, []string, bool) {
	var buf strings.Builder
	var registerImpls []string
	var clusterImpls []string
	hasPointers := false

	// Format the peripheral name
	peripheralName := periph.Name
	for _, p := range s.device.Peripherals.Elements {
		if p.DerivedFrom == periph.Name {
			peripheralName = periph.Name[:len(peripheralName)-1]
			break
		}
	}

	fmt.Fprintln(&buf, "struct {")
	offset := svd.Integer(0)

	// Sort the registers
	slices.SortStableFunc(periph.Registers.RegisterElements, func(a, b svd.RegisterElement) bool {
		return a.AddressOffset < b.AddressOffset
	})

	// Collect registers and cluster into the same list so they can be sorted by address offset
	var objs []svd.Addressable
	for _, cluster := range periph.Registers.ClusterElements {
		objs = append(objs, cluster)
	}

	for _, register := range periph.Registers.RegisterElements {
		objs = append(objs, register)
	}

	// Sort by address offset
	slices.SortStableFunc(objs, func(a, b svd.Addressable) bool {
		return a.GetAddressOffset() < b.GetAddressOffset()
	})

	for _, obj := range objs {
		switch obj := obj.(type) {
		case svd.ClusterElement:
			var clusterBuf strings.Builder
			clusterName := strings.ReplaceAll(obj.Name, "[%s]", "")

			offset += obj.Increment / s.device.WordSize

			if obj.AddressOffset >= offset {
				padding := obj.AddressOffset - offset
				if padding > periph.AddressBlock.Size {
					panic("padding exceeded the size of the peripheral")
				}

				// Insert padding bytes first
				if padding > 0 {
					fmt.Fprintf(&buf, "_ [%d]byte\n", padding)
				}
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
				fmt.Fprintf(&clusterBuf, "type S%s%s *struct{\n", peripheralName, clusterName)
			} else {
				fmt.Fprintf(&clusterBuf, "type S%s%s struct{\n", peripheralName, clusterName)
			}

			for _, register := range obj.Registers {
				registerName := strings.ReplaceAll(register.Name, "[%s]", "")
				registerTypename, registerImpl := s.generateRegisterType(peripheralName+clusterName, register)
				fmt.Fprintf(&clusterBuf, "%s %s\n", registerName, registerTypename)
				registerImpls = append(registerImpls, registerImpl)
				offset = obj.AddressOffset + (register.Size / s.device.WordSize)
			}
			fmt.Fprintln(&clusterBuf, "}\n")

			// Create the peripheral struct member
			fmt.Fprintf(&buf, "%s S%s%s\n", clusterName, peripheralName, clusterName)
			clusterImpls = append(clusterImpls, clusterBuf.String())
		case svd.RegisterElement:
			typename, _ := s.generateRegisterType(peripheralName, obj)

			if obj.AddressOffset >= offset {
				padding := obj.AddressOffset - offset
				if padding > periph.AddressBlock.Size {
					panic("padding exceeded the size of the peripheral")
				}

				// Insert padding bytes first
				if padding > 0 {
					fmt.Fprintf(&buf, "_ [%d]byte\n", padding)
				}
			}

			if obj.Count > 0 {
				fmt.Fprintf(&buf, "%s [%d]%s\n", strings.ReplaceAll(obj.Name, "[%s]", ""), obj.Count, typename)
			} else {
				fmt.Fprintf(&buf, "%s %s\n", obj.Name, typename)
			}
			offset = obj.AddressOffset + (obj.Size / s.device.WordSize)
		}
	}
	fmt.Fprint(&buf, "}")

	return buf.String(), registerImpls, clusterImpls, hasPointers
}

func (s *samgen) generateRegisterType(prefix string, register svd.RegisterElement) (string, string) {
	var buf strings.Builder
	typename := prefix + strings.ReplaceAll(register.Name, "[%s]", "")
	receiver := strings.ToLower(typename[0:1])

	// Declare the type
	fmt.Fprintf(&buf, "type %s %s\n\n", typename, typeForSize(register.Size))

	// Create enumerated types
	evMap := map[string]string{}
	for _, field := range register.Fields.Elements {
		if len(field.EnumeratedValues.Elements) > 0 {
			evTypename, evImpl := s.generateEnumeratedValuesType(typename, field.EnumeratedValues)

			// Map the field name to the enumerated type's typename
			evMap[field.Name] = evTypename

			// Write the implementation
			fmt.Fprintln(&buf, evImpl)
		}
	}

	// Create a setter/getter method for each field
	for _, field := range register.Fields.Elements {
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
			enumeratedType, hasEv := evMap[field.Name]
			if hasEv {
				fmt.Fprintf(&buf, "func (%s *%s) Get%s() %s {\n", receiver, typename, field.Name, enumeratedType)
			} else {
				returnType := typeForSize(register.Size)
				if field.BitWidth == 1 {
					returnType = "bool"
				}
				fmt.Fprintf(&buf, "func (%s *%s) Get%s() %s {\n", receiver, typename, field.Name, returnType)
			}

			if hasEv {
				fmt.Fprintf(&buf, "return %s(*%s)>>%d", enumeratedType, receiver, field.BitOffset)
			} else {
				exprPostfix := ""
				if field.BitWidth == 1 {
					exprPostfix = " != 0"
				}
				fmt.Fprintf(&buf, "return %s(*%s&(1<<%d))%s\n", typeForSize(register.Size), receiver, field.BitOffset, exprPostfix)
			}
			fmt.Fprintf(&buf, "}\n\n")
		}

		if access == "write-only" || access == "read-write" || access == "writeOnce" {
			enumeratedType, hasEv := evMap[field.Name]
			paramType := typeForBitWidth(field.BitWidth)
			if hasEv {
				paramType = enumeratedType
			}

			// Write the setter method
			fmt.Fprintf(&buf, "func (%s *%s) Set%s(value %s) {\n", receiver, typename, field.Name, paramType)
			if field.BitWidth == 1 && !hasEv {
				fmt.Fprintln(&buf, "if value {")
				fmt.Fprintf(&buf, "*%s = *%s|(1 << %d)\n", receiver, receiver, field.BitOffset)
				fmt.Fprintln(&buf, "} else {")
				fmt.Fprintf(&buf, "*%s = *%s& ^%s(1 << %d)\n", receiver, receiver, typename, field.BitOffset)
				fmt.Fprintln(&buf, "}")
			} else {
				fmt.Fprintf(&buf, "*%s = *%s& ^%s(%s << %d)\n", receiver, receiver, typename, allSet(field.BitWidth), field.BitOffset)
				fmt.Fprintf(&buf, "*%s = *%s|%s(value << %d)\n", receiver, receiver, typename, field.BitOffset)
			}

			fmt.Fprintf(&buf, "}\n\n")
		}
	}
	return typename, buf.String()
}

func (s *samgen) generateEnumeratedValuesType(prefix string, ev svd.EnumeratedValuesElement) (string, string) {
	var buf strings.Builder
	typename := prefix + ev.Name

	// Declare the type
	fmt.Fprintf(&buf, "type %s uint32\n\n", typename)
	fmt.Fprintln(&buf, "const (")
	// Create the constant values
	for _, value := range ev.Elements {
		if len(value.Description) > 0 {
			fmt.Fprintf(&buf, "// %s %s\n", typename+value.Name, value.Description)
		}
		fmt.Fprintf(&buf, "%s %s = %#x\n\n", typename+value.Name, typename, value.Value)
	}
	fmt.Fprintln(&buf, ")")

	return typename, buf.String()
}

func typeForSize(size svd.Integer) string {
	switch size {
	case 8:
		return "uint8"
	case 16:
		return "uint16"
	case 32:
		return "uint32"
	default:
		panic(fmt.Sprintf("unexpected size %d", size))
	}
}

func maxForSize(size svd.Integer) string {
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

func typeForBitWidth(width svd.Integer) string {
	switch width {
	case 1:
		return "bool"
	default:
		return "uint32"
	}
}

func allSet(bits svd.Integer) (result string) {
	for i := svd.Integer(0); i < bits; i++ {
		result += "1"
	}
	result = "0b" + result
	return
}
