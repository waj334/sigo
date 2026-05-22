package builder

import (
	"path/filepath"
)

// collectLinkerIncludePaths returns directories to search for #include in
// linker scripts. The current package's directory plus the directory of
// every .ld/.linker file we found (so a target.ld in the main package can
// #include "linker_chip_xi.ld" from a chip support package).
func collectLinkerIncludePaths(options linkOptions) []string {
	seen := map[string]struct{}{}
	var out []string
	add := func(d string) {
		if _, ok := seen[d]; ok {
			return
		}
		seen[d] = struct{}{}
		out = append(out, d)
	}
	for _, fname := range options.prog.Files[".ld"] {
		add(filepath.Dir(fname))
	}
	for _, fname := range options.prog.Files[".linker"] {
		add(filepath.Dir(fname))
	}
	return out
}

// collectLinkerDefines builds the -D map from BuildOptions and any
// TableGen-derived defaults (RAM_REGION, HEAP_REGION, STACK_REGION).
func collectLinkerDefines(options linkOptions) map[string]string {
	defines := map[string]string{}

	// From TableGen via the platform model — captured during platform parsing.
	// Whatever you store as the chip's default RAM/heap/stack target goes here.
	if options.ramRegion != "" {
		defines["RAM_REGION"] = options.ramRegion
	}
	if options.heapRegion != "" {
		defines["HEAP_REGION"] = options.heapRegion
	}
	if options.stackRegion != "" {
		defines["STACK_REGION"] = options.stackRegion
	}

	// User overrides from build flags (e.g., --ram=SRAM1).
	// These take precedence by being applied after the TableGen defaults.
	for k, v := range options.linkerDefines {
		defines[k] = v
	}

	return defines
}
