package main

import (
	"errors"
	"flag"
	"fmt"
	"omibyte.io/sigo/builder"
	"os"
	"path/filepath"
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
	flags := flag.NewFlagSet("build", flag.ExitOnError)
	flags.Usage = buildUsage

	// Add build args
	output := flags.String("o", ".", "output file")
	//procs := flags.Int("p", runtime.NumCPU(), "number of concurrent builds")

	// TODO: Implement dependency files for smarter make builds
	//createDependencyFiles := flags.Bool("MD", false, "create dependency files for Make")

	// Parse
	if err := flags.Parse(args); err != nil {
		println("Run 'sigo help build' for details")
		os.Exit(-1)
	}

	var packages []string
	if len(flags.Args()) == 0 {
		if cwd, err := os.Getwd(); err == nil {
			// Build the current directory by default
			packages = append(packages, cwd)
		} else {
			panic(err)
		}
	} else {
		// Convert the paths to absolute paths
		for _, arg := range flags.Args() {
			if filepath.IsLocal(arg) {
				path, _ := filepath.Abs(arg)
				packages = append(packages, path)
			} else {
				packages = append(packages, arg)
			}
		}
	}

	// Begin building the packages
	if err := builder.BuildPackages(packages, *output); err != nil {
		if errors.Is(err, builder.ErrParserError) {
			fmt.Println("Build error:", err)
		} else {
			fmt.Println("Compiler error:", err)
		}
	}
}
