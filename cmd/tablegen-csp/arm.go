package main

import (
	"cmp"
	"fmt"
	"go/format"
	"io"
	"slices"
	"strings"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generateArmInterruptVector(out io.Writer, variant *tablegen.Record) (int, error) {
	var builder strings.Builder
	var declBuilder strings.Builder
	var irqBuilder strings.Builder

	interrupts := variant.GetValueAsListOfDefs(constVariantFieldInterrupts)
	if len(interrupts) == 0 {
		return 0, nil
	}

	fmt.Fprint(&builder, `.syntax unified

.section .text.DefaultHandler
.global  DefaultHandler
.type    DefaultHandler, %function
DefaultHandler:
    wfe
    b    DefaultHandler
.size DefaultHandler, .-DefaultHandler

.macro IRQ handler
    .weak  \handler
    .set   \handler, DefaultHandler
.endm

.section .isr_vector, "a", %progbits
.global  __isr_vector
__isr_vector:
    .long __stack
`)

	// Sort the interrupts.
	slices.SortFunc(interrupts, func(a, b *tablegen.Record) int {
		linea := a.GetValueAsInt(constInterruptFieldLine)
		lineb := b.GetValueAsInt(constInterruptFieldLine)
		return cmp.Compare(linea, lineb)
	})

	maxHandlerLength := 0
	for _, interrupt := range interrupts {
		name := fmt.Sprintf("%sHandler", interrupt.GetValueAsString(constObjectFieldName))
		if len(name) > maxHandlerLength {
			maxHandlerLength = len(name)
		}
	}

	last := interrupts[0].GetValueAsInt(constInterruptFieldLine)
	for _, interrupt := range interrupts {
		interruptName := interrupt.GetValueAsString(constObjectFieldName)
		interruptName = formatGoIdentifier(formatCamelCase(strings.Split(interruptName, "_")...), true)
		interruptName = fmt.Sprintf("%sHandler", interruptName)
		description := interrupt.GetValueAsString(constObjectFieldDescription)
		line := interrupt.GetValueAsInt(constInterruptFieldLine)

		if last+1 < line {
			for i := last + 1; i < line; i++ {
				// Backfill with zeros.
				fmt.Fprintf(&declBuilder, "    .long 0\n")
			}
		}

		fmt.Fprintf(&declBuilder, "    .long %s\n", interruptName)

		if len(description) > 0 {
			fmt.Fprintf(&irqBuilder, "    IRQ %-*s // %s\n", 4+maxHandlerLength, interruptName, description)
		} else {
			fmt.Fprintf(&irqBuilder, "    IRQ %-*s\n", 4+maxHandlerLength, interruptName)
		}

		last = line
	}

	fmt.Fprintf(&builder, "%s\n", declBuilder.String())
	fmt.Fprintf(&builder, "%s\n", irqBuilder.String())

	return fmt.Fprint(out, builder.String())
}

func generateArmInterruptsSource(out io.Writer, series *tablegen.Record, variant *tablegen.Record) (int, error) {
	var builder strings.Builder

	seriesName := series.GetValueAsString(constObjectFieldName)
	packageName := strings.ToLower(formatGoIdentifier(strings.ToLower(seriesName), true))

	variantName := strings.ToLower(variant.GetValueAsString("name"))

	interrupts := variant.GetValueAsListOfDefs(constVariantFieldInterrupts)
	if len(interrupts) == 0 {
		return 0, nil
	}

	// Sort the interrupts.
	slices.SortFunc(interrupts, func(a, b *tablegen.Record) int {
		linea := a.GetValueAsInt(constInterruptFieldLine)
		lineb := b.GetValueAsInt(constInterruptFieldLine)
		return cmp.Compare(linea, lineb)
	})

	fmt.Fprintf(&builder, "//go:build %s && %s\n\n", packageName, variantName)
	fmt.Fprintf(&builder, "package %s\n\n", packageName)
	fmt.Fprintf(&builder, "import \"pkg.si-go.dev/chip/arm/cortexm/runtime\"\n\n")
	builder.WriteString("const (\n")

	for _, interrupt := range interrupts {
		interruptName := interrupt.GetValueAsString("name")
		interruptName = formatGoIdentifier(formatCamelCase(strings.Split(interruptName, "_")...), true)
		constName := fmt.Sprintf("Irq%s", interruptName)

		description := sanitizeDescription(interrupt.GetValueAsString("description"))
		line := interrupt.GetValueAsInt("line")

		trailingNewline := false
		if len(description) > 0 {
			fmt.Fprintf(&builder, "// %s %s\n", constName, description)
			trailingNewline = true
		}
		fmt.Fprintf(&builder, "%s runtime.Interrupt = %d\n", constName, line)

		if trailingNewline {
			fmt.Fprintf(&builder, "\n")
		}
	}

	builder.WriteString(")\n")

	srcStr := builder.String()
	src, err := format.Source([]byte(srcStr))
	if err != nil {
		panic(err)
	}

	return fmt.Fprint(out, string(src))
}
