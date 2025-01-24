package main

import (
	"fmt"
	"io"
	"strings"

	"omibyte.io/sigo/targets/device"
)

func writeLinkerScript(output io.StringWriter, v device.Variant) (int, error) {
	var builder strings.Builder

	builder.WriteString("MEMORY\n{\n")

	for _, memory := range v.Memories {
		fmt.Fprintf(&builder, "    MEM_%s (%s) : ORIGIN = %#x, LENGTH = %v\n",
			strings.ToUpper(memory.Identifier), translateAccess(memory.Flags), memory.Start, memory.Size)
	}

	builder.WriteString("}\n\n")

	// Write region aliases
	for _, memory := range v.Memories {
		if memory.Flags.IsSet(device.PrimaryFlash) {
			fmt.Fprintf(&builder, "REGION_ALIAS(\"FLASH\", MEM_%s);\n", strings.ToUpper(memory.Identifier))
		} else if memory.Flags.IsSet(device.PrimaryRam) {
			fmt.Fprintf(&builder, "REGION_ALIAS(\"RAM\", MEM_%s);\n", strings.ToUpper(memory.Identifier))
		}
	}

	var memories []device.Memory

	// Filter memories...
	for _, memory := range v.Memories {
		if memory.Flags.IsSet(device.PrimaryFlash) || memory.Flags.IsSet(device.PrimaryRam) {
			continue
		}
		memories = append(memories, memory)
	}

	if len(memories) > 0 {
		builder.WriteString("\nSECTIONS\n{\n\n")

		for _, memory := range memories {
			if memory.Flags.IsSet(device.PrimaryFlash) || memory.Flags.IsSet(device.PrimaryRam) {
				continue
			}

			name := fmt.Sprintf(".%sData", strings.ToLower(memory.Identifier))
			fmt.Fprintf(&builder, `    %[1]s :
    {
        *(%[1]s)
    } >MEM_%s

`,
				name, strings.ToUpper(memory.Identifier))
		}

		builder.WriteString("}\n\n")
	}

	builder.WriteString("__stack_size = 4K;\n")
	builder.WriteString("INCLUDE program.ld\n\n")

	return output.WriteString(builder.String())
}

func translateAccess(flags device.AttributeFlags) string {
	switch {
	case flags.IsSet(device.PrimaryFlash):
		return "rx"
	case flags.IsSet(device.Read, device.Write):
		return "rw"
	case flags.IsSet(device.Read):
		return "r"
	case flags.IsSet(device.Write):
		return "rw"
	default:
		// Not sure what to do here
		panic("none")
	}
}
