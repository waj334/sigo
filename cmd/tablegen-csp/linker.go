package main

import (
	"cmp"
	"fmt"
	"io"
	"slices"
	"strings"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generateLinkerScript(out io.Writer, variant *tablegen.Record) (int, error) {
	var builder strings.Builder

	memories := variant.GetValueAsListOfDefs(constVariantFieldMemories)
	if len(memories) == 0 {
		return 0, nil
	}

	// Sort memories by origin address.
	slices.SortFunc(memories, func(a, b *tablegen.Record) int {
		return cmp.Compare(a.GetValueAsInt(constRangeFieldOffset), b.GetValueAsInt(constRangeFieldOffset))
	})

	fmt.Fprintf(&builder, "MEMORY\n{\n")

	for _, memory := range memories {
		name := strings.ToUpper(memory.GetValueAsString(constObjectFieldName))
		description := memory.GetValueAsString(constObjectFieldDescription)
		origin := memory.GetValueAsInt(constRangeFieldOffset)
		lengthInKB := memory.GetValueAsInt(constRangeFieldWidth) / 1000
		executable := memory.GetValueAsBit(constMemoryRangeFieldExecutable)

		accessDef := memory.GetValueAsDef(constMemoryRangeFieldAccess)
		access := accessDef.GetValueAsString(constAccessModeValue)

		flags := strings.ToLower(access)
		if executable {
			flags += "x"
		}

		if len(description) > 0 {
			fmt.Fprintf(&builder, "\t%s (%s) : ORIGIN = %#x, LENGTH = %dK\t\t\t\t/* %s */\n", name, flags, origin, lengthInKB, description)
		} else {
			fmt.Fprintf(&builder, "\t%s (%s) : ORIGIN = %#x, LENGTH = %dK\n", name, flags, origin, lengthInKB)
		}
	}

	fmt.Fprintf(&builder, "}\n\n")

	stackSizeInKB := variant.GetValueAsInt(constVariantFieldStackSize) / 1000
	fmt.Fprintf(&builder, "__stack_size = %dK\n", stackSizeInKB)
	fmt.Fprintf(&builder, "INCLUDE program.ld\n\n")

	return fmt.Fprint(out, builder.String())
}
