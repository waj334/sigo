package main

import (
	"fmt"
	"go/format"
	"io"
	"strings"

	"omibyte.io/sigo/targets/device"
)

func writeInterruptsApi(output io.StringWriter, pkg string, irqPkg string, irqType string, variant device.Variant) (int, error) {
	var builder strings.Builder

	fmt.Fprintf(&builder, "package %s\n\n", strings.ToLower(formatSymbol(pkg, false)))
	fmt.Fprintf(&builder, "import \"%s\"\n\n", irqPkg)
	builder.WriteString("const (\n")

	for _, irq := range variant.Interrupts {
		symbol := formatSymbol(fmt.Sprintf("Irq_%s", irq.Identifier), true)
		if len(irq.Description) > 0 {
			fmt.Fprintf(&builder, "// %s %s\n", symbol, irq.Description)
		}
		fmt.Fprintf(&builder, "%s %s = %d\n", symbol, irqType, irq.Number)
	}

	builder.WriteString(")\n")

	srcStr := builder.String()
	src, err := format.Source([]byte(srcStr))
	if err != nil {
		panic(err)
	}

	return output.WriteString(string(src))
}
