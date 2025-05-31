package main

import (
	"fmt"
	"go/format"
	"io"
	"os"
	"slices"
	"strings"

	"pkg.si-go.dev/sigo/llvm/tablegen"
)

func generateSeriesInit(out io.Writer, series *tablegen.Record, requiredTags []string, optionalTags []string) (int, error) {
	var builder strings.Builder

	arch := series.GetValueAsDef(constSeriesFieldArchitecture)

	// Gather runtime packages that need to be imported.
	runtimePackages := series.GetValueAsListOfStrings(constSeriesRuntimePackages)
	runtimePackages = append(runtimePackages, arch.GetValueAsListOfStrings(constArchitectureRuntimePackages)...)

	if len(runtimePackages) == 0 {
		return 0, nil
	}

	// De-duplicate the imports.
	seen := make(map[string]struct{})
	deduped := make([]string, 0, len(runtimePackages))
	for _, pkg := range runtimePackages {
		if _, ok := seen[pkg]; !ok {
			seen[pkg] = struct{}{}
			deduped = append(deduped, pkg)
		}
	}
	runtimePackages = deduped

	// Sort the imports.
	slices.Sort(runtimePackages)

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
