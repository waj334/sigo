package main

import (
	"fmt"
	"io"
	"omibyte.io/sigo/targets/device"
	"slices"
	"strings"
)

func writeIsrVector(output io.StringWriter, v device.Variant) (int, error) {
	var builder strings.Builder

	// Write the preamble.
	builder.WriteString(`.syntax unified

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

	irqs := make([]device.Interrupt, len(v.Interrupts))
	copy(irqs, v.Interrupts)
	slices.SortFunc(irqs, func(a device.Interrupt, b device.Interrupt) int {
		return a.Number - b.Number
	})

	names := make([]string, len(irqs))

	maxHandlerLength := 0
	for i, irq := range irqs {
		name := fmt.Sprintf("%sHandler", formatSymbol(irq.Identifier, true))
		if len(name) > maxHandlerLength {
			maxHandlerLength = len(name)
		}
		names[i] = name
	}

	for i, irq := range irqs {
		if i > 0 {
			delta := (irq.Number - v.Interrupts[i-1].Number) - 1
			for ii := 0; ii < delta; ii++ {
				builder.WriteString("    .long 0\n")
			}
		}
		fmt.Fprintf(&builder, "    .long %s\n", names[i])
	}

	builder.WriteString("\n")

	for i, irq := range irqs {
		fmt.Fprintf(&builder, "    IRQ %-*s // %s\n", 4+maxHandlerLength, names[i], irq.Description)
	}

	builder.WriteString("\n")

	return output.WriteString(builder.String())
}
