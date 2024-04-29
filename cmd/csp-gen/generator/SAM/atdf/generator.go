package atdf

import (
	"fmt"
	"go/format"
	"golang.org/x/exp/slices"
	"io"
	"log"
	"math/bits"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"omibyte.io/sigo/cmd/csp-gen/atdf"
	"omibyte.io/sigo/cmd/csp-gen/generator"
	"omibyte.io/sigo/cmd/csp-gen/generator/SAM"
	"omibyte.io/sigo/cmd/csp-gen/types"
)

type _generator struct {
	def     *atdf.ATDF
	device  *atdf.DeviceElement
	clones  map[string][]string
	omitted map[string]struct{}
}

func NewGenerator(def *atdf.ATDF, device *atdf.DeviceElement) generator.Generator {
	return &_generator{
		def:     def,
		device:  device,
		clones:  map[string][]string{},
		omitted: map[string]struct{}{},
	}
}

func (g *_generator) Generate(out string) error {
	// Create the output directory for the chip
	outputDir := filepath.Join(out, "chip", strings.ToLower(g.device.Name))
	if err := os.MkdirAll(outputDir, 0750); err != nil {
		return err
	}

	// Generate package init
	if err := SAM.GenerateInit(g.device.Name, outputDir); err != nil {
		return err
	}

	// Generate the linker script
	if err := g.generateLinkerScript(outputDir); err != nil {
		return err
	}

	// Generate IRQ handlers
	if err := g.generateISR(outputDir); err != nil {
		return err
	}

	// Generate the CSP
	if err := g.generateCSP(out); err != nil {
		return err
	}

	return nil
}

func (g *_generator) generateLinkerScript(out string) (err error) {
	var w strings.Builder

	fmt.Fprintln(&w, "MEMORY")
	fmt.Fprintln(&w, "{")

	// Write flash sections
	for _, spaces := range g.device.AddressSpaces.Elements {
		for _, segment := range spaces.MemorySegments {
			if segment.Type == "flash" {
				fmt.Fprintf(&w, "\t%s (%s) : ORIGIN = %#x, LENGTH = %#x\n", segment.Name, strings.ToLower(segment.RW), segment.Start, segment.Size)
			}
		}
	}

	// Write RAM sections
	for _, spaces := range g.device.AddressSpaces.Elements {
		for _, segment := range spaces.MemorySegments {
			if segment.Type == "ram" {
				name := segment.Name

				// RAM has different names smh
				if strings.Contains(name, "HSRAM") {
					name = strings.ReplaceAll(segment.Name, "HSRAM", "RAM")
				}

				if name == "HMCRAMC0" {
					//SAMD21
					name = "RAM"
				}

				fmt.Fprintf(&w, "\t%s (%s) : ORIGIN = %#x, LENGTH = %#x\n", name, strings.ToLower(segment.RW), segment.Start, segment.Size)
			}
		}
	}

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

func (g *_generator) generateISR(out string) (err error) {
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
`)

	// Collect all the interrupts
	interrupts := map[types.Integer]atdf.InterruptElement{}
	irqMaxValue := types.Integer(0)
	irqLowValue := types.Integer(0)
	for _, irq := range g.device.Interrupts.Elements {
		interrupts[irq.Index] = irq
		if irq.Index > irqMaxValue {
			irqMaxValue = irq.Index
		}

		if irq.Index < irqLowValue {
			irqLowValue = irq.Index
		}
	}

	// Fill the remainder of the vector table with the peripheral interrupt handlers
	for i := irqLowValue; i < irqMaxValue; i++ {
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
		if len(irq.Caption) > 0 {
			comment = "\t\t\t\t// " + irq.Caption
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

func (g *_generator) generateCSP(out string) (err error) {
	var w, initializers, modules strings.Builder

	// Write the preamble to the file
	SAM.WritePreamble(&w, g.device.Name, "chip")

	// Write required imports
	fmt.Fprintln(&w, "import (")
	fmt.Fprintln(&w, `_ "runtime/arm/cortexm"`)
	fmt.Fprintln(&w, `"unsafe"`)
	fmt.Fprintln(&w, `"volatile"`)
	fmt.Fprintln(&w, ")\n\n")

	// Generate types first
	for _, module := range g.def.Modules.Elements {
		g.generateModule(&modules, module)
	}

	// Generate initializers
	g.generateInitializers(&initializers)

	w.WriteString(initializers.String())
	w.WriteString(modules.String())

	fname := strings.ToLower(strings.ToLower(g.device.Name) + ".go")

	// Format the final output
	var buf []byte
	src := w.String()
	buf, err = format.Source([]byte(src))
	if err != nil {
		f, err := os.Create(filepath.Join(out, "chip", fname))
		if err != nil {
			return err
		}
		f.Write([]byte(w.String()))
		f.Close()
		return fmt.Errorf("error formatting %s: %v", fname, err)
	}

	// Write the contents to the file
	f, err := os.Create(filepath.Join(out, "chip", fname))
	if err != nil {
		return err
	}
	f.Write(buf)
	f.Close()

	return nil
}

func (g *_generator) generateInitializers(w io.Writer) {
	fmt.Fprintf(w, "var (\n")

	type entry struct {
		typename string
		offset   types.Integer
	}

	merges := map[string][]entry{}

	for _, module := range g.device.Peripherals.Modules {
		for _, group := range module.Instance.RegisterGroups {
			if _, ok := g.omitted[group.NameInModule]; ok {
				continue
			}

			instances := []string{group.NameInModule + "_TYPE"}
			if clones, ok := g.clones[group.NameInModule]; ok {
				instances = clones
			}

			for _, instance := range instances {
				varName := group.Name
				if strings.Contains(instance, ".") {
					parts := strings.Split(instance, ".")
					varName = parts[0]
					instance = parts[1]
				}

				merges[varName] = append(merges[varName], entry{
					typename: instance,
					offset:   group.Offset,
				})
			}
		}
	}

	for varName, e := range merges {
		if len(e) > 1 {
			// Sort by address
			slices.SortFunc(e, func(a, b entry) int {
				return int(a.offset - b.offset)
			})

			typename := e[0].typename
			fmt.Fprintf(w, "%s = [%d]*%s{\n", varName, len(e), typename)
			for _, ee := range e {
				fmt.Fprintf(w, "(*%s)(unsafe.Pointer(uintptr(%#x))),\n", ee.typename, ee.offset)
			}
			fmt.Fprintf(w, "}\n")
		} else {
			fmt.Fprintf(w, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n", varName, e[0].typename, e[0].offset)
		}
	}

	fmt.Fprintf(w, ")\n\n")
}

func (g *_generator) generateModule(w io.Writer, module atdf.ModuleElement) {
	// Create the peripheral types
	for _, group := range module.RegisterGroups {
		if len(group.Registers) == 0 && len(group.Groups) == 0 {
			// Track this omission so no initializer is created for it
			g.omitted[group.Name] = struct{}{}

			// Skip empty groups
			continue
		}

		// Is this group referenced by another group?
		externalReference := false
		for _, otherGroup := range module.RegisterGroups {
			if groupReferences(group.Name, otherGroup) {
				externalReference = true
				break
			}
		}

		if !externalReference {
			if len(group.Modes) > 0 {
				for _, mode := range group.Modes {
					if len(group.Registers) > 0 {
						typename := fmt.Sprintf("%s_%s_TYPE", group.Name, mode.Name)
						varname := fmt.Sprintf("%s_%s", group.Name, mode.Name)
						fmt.Fprintf(w, "type %s struct {\n", typename)
						generateRegisterGroupStructBody(w, module, group, mode.Name)
						fmt.Fprintf(w, "}\n\n")
						// Track this clone of the peripheral
						g.clones[module.Name] = append(g.clones[module.Name], varname+"."+typename)
					}
				}
			} else {
				// Create the peripheral struct type
				fmt.Fprintf(w, "type %s_TYPE struct {\n", group.Name)
				generateRegisterGroupStructBody(w, module, group, "")
				fmt.Fprintf(w, "}\n\n")
			}
		} else {
			name := fmt.Sprintf("%s_%s_TYPE", module.Name, group.Name)
			fmt.Fprintf(w, "type %s struct{\n", name)
			generateRegisterGroupStructBody(w, module, group, "")
			fmt.Fprintf(w, "}\n\n")
		}

		// Create the register types
		for _, register := range group.Registers {
			registerTypeName := registerName(module, register)
			intType := typeForSize(register.Size)
			fmt.Fprintf(w, "type %s %s", registerTypeName, intType)

			if len(register.Caption) > 0 {
				fmt.Fprintf(w, " // %s", register.Caption)
			}
			fmt.Fprintln(w)

			// Generate the functions for this register
			for _, bitfield := range register.BitFields {
				if len(bitfield.Values) > 0 {
					// Find the value group
					valueGroup := module.FindValueGroup(bitfield.Values)
					if valueGroup == nil {
						panic("value group not found")
					}

					typeName := fmt.Sprintf("%s_%s", registerTypeName, bitfield.Name)

					// Create the type for this value group
					fmt.Fprintf(w, "type %s %s\n", typeName, intType)

					// Create the values
					fmt.Fprintf(w, "const (\n")
					for _, value := range valueGroup.Elements {
						fmt.Fprintf(w, "%s_%s_%s %s = %#x", registerTypeName, bitfield.Name, value.Name, typeName, value.Value)
						if len(value.Caption) > 0 {
							fmt.Fprintf(w, " // %s", value.Caption)
						}
						fmt.Fprintln(w)
					}
					fmt.Fprintf(w, ")\n\n")
				}

				generateBitfieldFuncs(w, module, group, register, bitfield)
			}
		}
		fmt.Fprintf(w, "\n")
	}
}

func generateRegisterGroupStructBody(w io.Writer, module atdf.ModuleElement, group atdf.ModuleRegisterGroupElement, mode string) {
	var l []atdf.Offsetable

	// Create a homogeneous list of the registers and register groups
	for _, r := range group.Registers {
		l = append(l, r)
	}

	for _, g := range group.Groups {
		l = append(l, g)
	}

	// Sort the list by offset
	slices.SortFunc(l, func(a, b atdf.Offsetable) int {
		return int(a.Offset() - b.Offset())
	})

	if len(l) == 0 {
		return
	}

	offset := types.Integer(0)
	for _, obj := range l {
		switch register := obj.(type) {
		case atdf.RegisterElement:
			if len(mode) > 0 && len(register.Mode) > 0 && register.Mode != mode {
				// The register belongs to a different mode. Skip it
				continue
			}

			if offset != register.Offset() {
				n := register.Offset() - offset
				// Insert padding bytes
				fmt.Fprintf(w, "_ [%d]byte\n", n)
				offset += n
			}

			// Create struct field
			registerTypeName := registerName(module, register)
			if register.Count > 0 {
				// Create a fixed-size array representing this register
				fmt.Fprintf(w, "%s [%d]%s", register.Name, register.Count, registerTypeName)
				offset = register.Offset() + (register.Size * register.Count)
			} else {
				fmt.Fprintf(w, "%s %s", register.Name, registerTypeName)
				offset = register.Offset() + register.Size
			}

			if len(register.Caption) > 0 {
				fmt.Fprintf(w, " // %s", register.Caption)
			}
			fmt.Fprintln(w)
		case atdf.ModuleRegisterGroupElement:
			typename := fmt.Sprintf("%s_%s_TYPE", module.Name, register.NameInModule)
			if register.Count > 0 {
				// Create a fixed-size array of the group type
				fmt.Fprintf(w, "%s [%d]%s\n", register.Name, register.Count, typename)
				offset = register.Offset() + (register.Size * register.Count)
			} else {
				fmt.Fprintf(w, "%s %s\n", register.Name, typename)
				offset = register.Offset() + register.Size
			}
		}
	}
	if offset < group.Size {
		n := group.Size - offset
		// Insert padding bytes
		fmt.Fprintf(w, "_ [%d]byte\n", n)
	}
}

func generateBitfieldFuncs(w io.Writer, module atdf.ModuleElement, group atdf.ModuleRegisterGroupElement, register atdf.RegisterElement, bitfield atdf.BitFieldElement) {
	typename, offset := typeForMask(bitfield.Mask)
	intType := typeForSize(register.Size)
	registerTypeName := registerName(module, register)

	if len(bitfield.Values) > 0 {
		typename = fmt.Sprintf("%s_%s", registerTypeName, bitfield.Name)
	}

	alts := []string{""}
	if len(bitfield.Modes) > 0 {
		alts = strings.Split(bitfield.Modes, " ")
	}

	for _, alt := range alts {
		if strings.Contains(strings.ToLower(register.RW), "r") {
			fmt.Fprintf(w, "func (reg *%s) Get%s%s() %s {", registerTypeName, alt, cleanIdentifier(bitfield.Name), typename)
			fmt.Fprintf(w, "v := volatile.Load%s((*%s)(reg))\n", strings.Title(intType), intType)
			if typename == "bool" {
				fmt.Fprintf(w, "return v&(1<<%d) != 0\n", offset)
			} else {
				fmt.Fprintf(w, "return %s(v&%#x) >> %d\n", typename, bitfield.Mask, offset)
			}
			fmt.Fprintf(w, "}\n\n")
		}

		if strings.Contains(strings.ToLower(register.RW), "w") {
			if typename == "bool" {
				fmt.Fprintf(w, "func (reg *%s) Set%s%s(enable bool)  {\n", registerTypeName, alt, cleanIdentifier(bitfield.Name))
				fmt.Fprintf(w, "v := volatile.Load%s((*%s)(reg))\n", strings.Title(intType), intType)
				fmt.Fprintf(w, "if enable {\n")
				fmt.Fprintf(w, "v |= 1 << %d\n", offset)
				fmt.Fprintf(w, "} else {\n")
				fmt.Fprintf(w, "v &^= 1 << %d\n", offset)
				fmt.Fprintf(w, "}\n")
			} else {
				fmt.Fprintf(w, "func (reg *%s) Set%s%s(value %s)  {\n", registerTypeName, alt, cleanIdentifier(bitfield.Name), typename)
				fmt.Fprintf(w, "v := volatile.Load%s((*%s)(reg))\n", strings.Title(intType), intType)
				fmt.Fprintf(w, "v &^= %#x\n", bitfield.Mask)              // Unset the respective bits.
				fmt.Fprintf(w, "v |= %s(value) << %d\n", intType, offset) // Set the respective bits to the specified value.
			}
			fmt.Fprintf(w, "volatile.Store%s((*%s)(reg), v)\n", strings.Title(intType), intType)
			fmt.Fprintf(w, "}\n\n")
		}
	}
}

func registerName(module atdf.ModuleElement, register atdf.RegisterElement) string {
	if len(register.Mode) > 0 {
		return fmt.Sprintf("%s_%s_%s_REG", module.Name, register.Mode, register.Name)
	}
	return fmt.Sprintf("%s_%s_REG", module.Name, register.Name)
}

func groupReferences(name string, group atdf.ModuleRegisterGroupElement) bool {
	if group.NameInModule == name {
		return true
	} else if len(group.Groups) > 0 {
		for _, subgroup := range group.Groups {
			if groupReferences(name, subgroup) {
				return true
			}
		}
	}

	return false
}

func typeForSize(size types.Integer) (result string) {
	switch {
	case size <= 1:
		result = "uint8"
	case size <= 2:
		result = "uint16"
	case size <= 4:
		result = "uint32"
	default:
		log.Panicf("supported type size %d", size)
	}
	return result
}

func typeForMask(mask types.Integer) (goType string, offset int) {
	numBits := bits.OnesCount64(uint64(mask))
	offset = bits.TrailingZeros64(uint64(mask))
	switch {
	case numBits <= 1:
		goType = "bool"
	case numBits <= 8:
		goType = "uint8"
	case numBits <= 16:
		goType = "uint16"
	default:
		goType = "uint32"
	}
	return
}

func cleanIdentifier(ident string) string {
	re := regexp.MustCompile(`([a-zA-Z0-9]$|[a-zA-Z0-9][_a-zA-Z0-9]*[a-zA-Z0-9])`)
	cleanStr := re.FindStringSubmatch(ident)
	return cleanStr[0]
}
