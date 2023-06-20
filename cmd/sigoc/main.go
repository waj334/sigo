package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"omibyte.io/sigo/builder"
	"omibyte.io/sigo/compiler"
)

var (
	command string
)

func init() {
	flag.CommandLine.Usage = usage

	// Parse the command line arguments
	flag.Parse()
}

func usage() {
	print(`SiGo is a compiler for the Go language targeting embedded systems.

Usage:

        sigo <command> [arguments]

The commands are:

	build       compile packages and dependencies
	clean       remove object files and cached files
	env	    print sigo environment information
	version     print sigo version
	help        display this help prompt

Use "sigo help <topic> for more information about that topic."

`)
}

func buildUsage() {
	println("usage: sigo build [-o output] [build flags] [packages]")
}

func main() {
	// Process the arguments
	command = flag.Arg(0)

	switch command {
	case "build":
		build(flag.Args()[1:])
	case "clean":
		panic("Not implemented")
	case "env":
		panic("Not implemented")
	case "version":
		panic("Not implemented")
	case "":
		fallthrough
	case "help":
		command = ""
		usage()
	default:
		fmt.Printf("sigo %s: unknown command\n", command)
		println("Run 'sigo help' for usage")
		os.Exit(-1)
	}
}

func build(args []string) {
	buildFlags := flag.NewFlagSet("build", flag.ExitOnError)

	// Add build args
	output := buildFlags.String("o", ".", "output file")
	verbose := buildFlags.String("verbose", "", "verbosity level")
	debug := buildFlags.Bool("g", false, "generate debug information")
	dumpOnVerError := buildFlags.Bool("dump-verify", false, "dump IR upon verification error")
	dumpIR := buildFlags.Bool("dump-ir", false, "dump the IR")
	tags := buildFlags.String("tags", "", "build tags")
	useCTypeNames := buildFlags.Bool("ctypenames", false, "use C type names for primitives in debug information")
	numJobs := buildFlags.Int("j", runtime.NumCPU(), "number of concurrent builds")
	optimize := buildFlags.String("O", "0", "optimization level")

	// TODO: Implement dependency files for smarter make builds
	//createDependencyFiles := flags.Bool("MD", false, "create dependency files for Make")

	// Parse
	if err := buildFlags.Parse(args); err != nil {
		println("Run 'sigo help build' for details")
		os.Exit(-1)
	}

	// Get the current working directory
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	builderOptions := builder.Options{
		Output:            *output,
		DumpOnVerifyError: *dumpOnVerError,
		DumpIR:            *dumpIR,
		Environment:       builder.Environment(),
		CompilerVerbosity: compiler.Debug,
		GenerateDebugInfo: *debug,
		BuildTags:         strings.Split(*tags, ","),
		CTypeNames:        *useCTypeNames,
		NumJobs:           *numJobs,
		Optimization:      *optimize,
	}

	if len(buildFlags.Args()) == 0 {
		// Build the current directory by default
		builderOptions.Packages = append(builderOptions.Packages, cwd)
	} else {
		// Convert the paths to relative paths
		for _, arg := range buildFlags.Args() {
			if filepath.IsAbs(arg) {
				path, _ := filepath.Rel(cwd, arg)
				builderOptions.Packages = append(builderOptions.Packages, path)
			} else {
				builderOptions.Packages = append(builderOptions.Packages, arg)
			}
		}
	}

	// Determine output verbosity
	switch strings.ToLower(*verbose) {
	case "", "verbose":
		builderOptions.CompilerVerbosity = compiler.Verbose
	case "quiet":
		builderOptions.CompilerVerbosity = compiler.Quiet
	case "info":
		builderOptions.CompilerVerbosity = compiler.Info
	case "warning":
		builderOptions.CompilerVerbosity = compiler.Warning
	case "debug":
		builderOptions.CompilerVerbosity = compiler.Debug
	default:
		println("Unknown output verbosity mode. Defaulting to \"quiet\"")
		builderOptions.CompilerVerbosity = compiler.Quiet
	}

	// Begin building the packages
	if err = builder.BuildPackages(context.Background(), builderOptions); err != nil {
		if errors.Is(err, builder.ErrParserError) {
			fmt.Println("Build error:", err)
		} else {
			fmt.Println("Compiler error:", err)
		}
	}
}
