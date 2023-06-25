package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"omibyte.io/sigo/builder"
	"omibyte.io/sigo/compiler"
)

var (
	buildOpts = struct {
		output   string
		verbose  string
		debug    bool
		dumpIR   bool
		tags     string
		ctypes   bool
		jobs     int
		optimize string
	}{}

	buildCmd = &cobra.Command{
		Use:   "build",
		Short: "Build a package or .go file",
		Long:  "Build a package or .go file and output a flashable image for a target system",
		Run: func(cmd *cobra.Command, args []string) {
			// Get the current working directory
			cwd, err := os.Getwd()
			if err != nil {
				panic(err)
			}

			builderOptions := builder.Options{
				Output:            buildOpts.output,
				DumpIR:            buildOpts.dumpIR,
				Environment:       builder.Environment(),
				CompilerVerbosity: compiler.Debug,
				GenerateDebugInfo: buildOpts.debug,
				BuildTags:         strings.Split(buildOpts.tags, ","),
				CTypeNames:        buildOpts.ctypes,
				NumJobs:           buildOpts.jobs,
				Optimization:      buildOpts.optimize,
			}

			if len(cmd.Flags().Args()) == 0 {
				// Build the current directory by default
				builderOptions.Packages = append(builderOptions.Packages, cwd)
			} else {
				// Convert the paths to relative paths
				for _, arg := range cmd.Flags().Args() {
					if filepath.IsAbs(arg) {
						path, _ := filepath.Rel(cwd, arg)
						builderOptions.Packages = append(builderOptions.Packages, path)
					} else {
						builderOptions.Packages = append(builderOptions.Packages, arg)
					}
				}
			}

			// Determine output verbosity
			switch strings.ToLower(buildOpts.verbose) {
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
		},
	}
)

func init() {
	buildCmd.Flags().StringVarP(&buildOpts.output, "output", "o", ".", "output file")
	buildCmd.Flags().StringVarP(&buildOpts.verbose, "verbose", "v", "", "verbosity level")
	buildCmd.Flags().BoolVarP(&buildOpts.debug, "debug", "g", false, "generate debug information")
	buildCmd.Flags().BoolVar(&buildOpts.dumpIR, "dump-ir", false, "dump the IR")
	buildCmd.Flags().StringVarP(&buildOpts.tags, "tags", "t", "", "build tags")
	buildCmd.Flags().BoolVar(&buildOpts.ctypes, "ctypenames", false, "use C type names for primitives in debug information")
	buildCmd.Flags().IntVarP(&buildOpts.jobs, "jobs", "j", runtime.NumCPU(), "number of concurrent builds")
	buildCmd.Flags().StringVarP(&buildOpts.optimize, "opt", "O", "0", "optimization level")
}
