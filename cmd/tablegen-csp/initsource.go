package main

import (
	"fmt"
	"go/format"
	"io"
	"os"
	"strings"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generateSeriesInit(out io.Writer, series *tablegen.Record, requiredTags []string, optionalTags []string) (int, error) {
	var builder strings.Builder

	arch := series.GetValueAsDef(constSeriesFieldArchitecture)
	runtimePackages := arch.GetValueAsListOfStrings(constArchitectureRuntimePackages)

	seriesName := series.GetValueAsString(constObjectFieldName)
	packageName := strings.ToLower(formatGoIdentifier(strings.ToLower(seriesName), true))

	if len(requiredTags) > 0 || len(optionalTags) > 0 {
		fmt.Fprintf(&builder, "//go:build %s\n\n", tagsString(requiredTags, optionalTags))
	}

	fmt.Fprintf(&builder, "package %s\n\n", packageName)
	fmt.Fprintf(&builder, "import (\n")

	for _, pkg := range runtimePackages {
		fmt.Fprintf(&builder, "_ \"%s\"\n", pkg)
	}

	fmt.Fprintf(&builder, ")\n\n")

	// Format the final output.
	srcStr := builder.String()
	src, err := format.Source([]byte(srcStr))
	if err != nil {
		fmt.Fprintf(os.Stderr, "*** START ***\n%s\n*** END ***\n", srcStr)
		return 0, err
	}
	return fmt.Fprint(out, string(src))
}
