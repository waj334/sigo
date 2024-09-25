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
)

var (
	buildOpts = struct {
		output      string
		verbose     string
		debug       bool
		dumpIR      bool
		cpu         string
		float       string
		tags        string
		ctypes      bool
		jobs        int
		optimize    string
		stackSize   int
		keepWorkDir bool
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

			// Check the target CPU string
			if len(buildOpts.cpu) == 0 {
				fmt.Fprintln(os.Stderr, "no target CPU specified.")
				fmt.Fprintln(os.Stderr, "Example:")
				fmt.Fprintln(os.Stderr, "-cpu=\"atsamd21g18a\"")
				fmt.Fprintln(os.Stderr, `--cpu=samd21 --tags=atsamd21g18a,other,tags,...`)
				cmd.Help()
				return
			}

			// Check the float string
			if buildOpts.float != "softfp" && buildOpts.float != "hardfp" {
				fmt.Fprintln(os.Stderr, "invalid floating-point mode\"", buildOpts.float, "\" specified")
				cmd.Help()
				return
			}

			env, err := builder.Environment()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Toolchain error: %v", err)
				return
			}

			builderOptions := builder.Options{
				Output:      buildOpts.output,
				DumpIR:      buildOpts.dumpIR,
				Environment: env,
				//CompilerVerbosity: compiler.Debug,
				GenerateDebugInfo: buildOpts.debug,
				Cpu:               buildOpts.cpu,
				Float:             buildOpts.float,
				CTypeNames:        buildOpts.ctypes,
				NumJobs:           buildOpts.jobs,
				Optimization:      buildOpts.optimize,
				StackSize:         buildOpts.stackSize,
				KeepWorkDir:       buildOpts.keepWorkDir,
			}

			if len(buildOpts.tags) > 0 {
				builderOptions.BuildTags = strings.Split(buildOpts.tags, ",")
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
			/*switch strings.ToLower(buildOpts.verbose) {
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
			}*/

			// Begin building the packages
			if err = builder.BuildPackages(context.Background(), builderOptions); err != nil {
				if errors.Is(err, builder.ErrParserError) {
					fmt.Fprintf(os.Stderr, "Build error: %v", err)
				} else {
					fmt.Fprintf(os.Stderr, "Compiler error: %v", err)
				}
				return
			}
		},
	}
)

func init() {
	buildCmd.Flags().StringVarP(&buildOpts.output, "output", "o", ".", "output file")
	buildCmd.Flags().StringVarP(&buildOpts.verbose, "verbose", "v", "", "verbosity level")
	buildCmd.Flags().StringVar(&buildOpts.cpu, "cpu", "", "target cpu")
	buildCmd.Flags().StringVar(&buildOpts.float, "float", "softfp", "floating-point mode (=softfp, =hardfp")
	buildCmd.Flags().BoolVarP(&buildOpts.debug, "debug", "g", false, "generate debug information")
	buildCmd.Flags().BoolVar(&buildOpts.dumpIR, "dump-ir", false, "dump the IR")
	buildCmd.Flags().StringVarP(&buildOpts.tags, "tags", "t", "", "build tags")
	buildCmd.Flags().BoolVar(&buildOpts.ctypes, "ctypenames", false, "use C type names for primitives in debug information")
	buildCmd.Flags().IntVarP(&buildOpts.jobs, "jobs", "j", runtime.NumCPU(), "number of concurrent builds")
	buildCmd.Flags().StringVarP(&buildOpts.optimize, "opt", "O", "0", "optimization level")
	buildCmd.Flags().IntVarP(&buildOpts.stackSize, "stack-size", "s", 2048, "stack size of each goroutine")
	buildCmd.Flags().BoolVar(&buildOpts.keepWorkDir, "work", false, "do not delete the work directory upon build")
}
