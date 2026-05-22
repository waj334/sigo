package main

import (
	"cmp"
	"fmt"
	"io"
	"slices"
	"strings"
	"text/tabwriter"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generateLinkerScript(out io.Writer, variant *tablegen.Record) (int, error) {
	memories := variant.GetValueAsListOfDefs(constVariantFieldMemories)
	if len(memories) == 0 {
		return 0, nil
	}

	// Sort memories by origin address for readable output.
	slices.SortFunc(memories, func(a, b *tablegen.Record) int {
		return cmp.Compare(
			a.GetValueAsInt(constRangeFieldOffset),
			b.GetValueAsInt(constRangeFieldOffset),
		)
	})

	var builder strings.Builder
	builder.WriteString("MEMORY\n{\n")

	// Use tabwriter for the MEMORY block to align columns. The leading tab
	// in each row indents inside the braces.
	tw := tabwriter.NewWriter(&builder, 0, 4, 1, ' ', 0)
	for _, memory := range memories {
		name := strings.ToUpper(memory.GetValueAsString(constObjectFieldName))
		description := memory.GetValueAsString(constObjectFieldDescription)
		origin := memory.GetValueAsInt(constRangeFieldOffset)
		executable := memory.GetValueAsBit(constMemoryRangeFieldExecutable)

		accessDef := memory.GetValueAsDef(constMemoryRangeFieldAccess)
		access := accessDef.GetValueAsString(constAccessModeValue)

		flags := strings.ToLower(access)
		if executable {
			flags += "x"
		}

		length := memory.GetValueAsInt(constRangeFieldWidth)
		var lengthStr string
		switch {
		case length >= 1024*1024 && length%(1024*1024) == 0:
			lengthStr = fmt.Sprintf("%dM", length/(1024*1024))
		case length >= 1024 && length%1024 == 0:
			lengthStr = fmt.Sprintf("%dK", length/1024)
		default:
			lengthStr = fmt.Sprintf("%d", length)
		}

		if len(description) > 0 {
			fmt.Fprintf(tw, "\t%s\t(%s)\t: ORIGIN = %#010x,\tLENGTH = %s\t/* %s */\n",
				name, flags, origin, lengthStr, description)
		} else {
			fmt.Fprintf(tw, "\t%s\t(%s)\t: ORIGIN = %#010x,\tLENGTH = %s\t\n",
				name, flags, origin, lengthStr)
		}
	}

	// Memory section extensions.
	fmt.Fprintf(tw, "#ifdef EXTRA_MEMORY_REGIONS\n")
	fmt.Fprintf(tw, "\tEXTRA_MEMORY_REGIONS\n")
	fmt.Fprintf(tw, "#endif\n")
	tw.Flush()
	builder.WriteString("}\n\n")

	// Emit region alias defaults from variant declarations.
	writeRegionDefault(&builder, "TEXT_REGION",
		variant.GetValueAsString(constVariantFieldDefaultTextRegion))
	writeRegionDefault(&builder, "RAM_REGION",
		variant.GetValueAsString(constVariantFieldDefaultRAMRegion))
	writeRegionDefault(&builder, "HEAP_REGION",
		variant.GetValueAsString(constVariantFieldDefaultHeapRegion))
	writeRegionDefault(&builder, "STACK_REGION",
		variant.GetValueAsString(constVariantFieldDefaultStackRegion))

	builder.WriteString("\n")

	// Aliases. Always emit — they fall through to the chip's default
	// defines if the user didn't override.
	builder.WriteString(`REGION_ALIAS("TEXT",  TEXT_REGION);` + "\n")
	builder.WriteString(`REGION_ALIAS("RAM",   RAM_REGION);` + "\n")
	builder.WriteString(`REGION_ALIAS("HEAP",  HEAP_REGION);` + "\n")
	builder.WriteString(`REGION_ALIAS("STACK", STACK_REGION);` + "\n\n")

	// Emit SECTIONS block. One section per memory range, multi-line for
	// readability rather than column-aligned (sections are too wide).
	builder.WriteString("SECTIONS\n{\n")
	for i, memory := range memories {
		name := strings.ToUpper(memory.GetValueAsString(constObjectFieldName))
		lower := strings.ToLower(name)
		align := memory.GetValueAsInt(constMemoryRangeFieldAlign)

		if i > 0 {
			builder.WriteByte('\n')
		}
		fmt.Fprintf(&builder,
			"\t.%[1]s (NOLOAD) : ALIGN(%[3]d)\n"+
				"\t{\n"+
				"\t\t__%[1]s_start = .;\n"+
				"\t\tKEEP(*(.%[1]s))\n"+
				"\t\tKEEP(*(.%[1]s.*))\n"+
				"\t\t. = ALIGN(%[3]d);\n"+
				"\t\t__%[1]s_end = .;\n"+
				"\t} > %[2]s\n",
			lower, name, align)
	}
	builder.WriteString("}\n\n")

	stackSizeInKB := variant.GetValueAsInt(constVariantFieldStackSize) / 1000
	fmt.Fprintf(&builder, "__stack_size = %dK;\n", stackSizeInKB)
	builder.WriteString("#include <program.ld>\n")

	return fmt.Fprint(out, builder.String())
}

// writeRegionDefault emits an #ifndef-guarded default for a region macro.
// If region is empty (unset), no default is written, so the linker will
// require the user to supply one or fail with a clear error.
func writeRegionDefault(b *strings.Builder, macro, region string) {
	if region == "" {
		return
	}
	fmt.Fprintf(b, "#ifndef %s\n#define %s %s\n#endif\n", macro, macro, region)
}
