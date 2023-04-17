package builder

import (
	"context"
	"errors"
	"go/token"
	"go/types"
	"omibyte.io/sigo/compiler"
	"omibyte.io/sigo/llvm"
	"os"
	"path"
	"path/filepath"
)

func BuildPackages(packageDirectories []string, output string) error {
	// Check output path with respect to the number of input packages
	if info, err := os.Stat(output); err == nil && !info.IsDir() && len(packageDirectories) > 1 {
		// Output must be a path if multiple packages were specified
		return ErrUnexpectedOutputPath
	}

	// Build each package
	for _, directory := range packageDirectories {
		info, err := os.Stat(directory)
		if err != nil {
			return errors.Join(ErrParserError, err)
		} else if info.IsDir() {
			return Build(directory, output, Environment())
		} else {
			// TODO: Allow a mix of package directories and individual .go files?
			panic("Not implemented")
		}
	}

	return nil
}

func Build(packageDirectory, output string, env Env) error {
	fset := token.NewFileSet()
	config := types.Config{}

	// Create a new program
	prog := Program{
		config:     config,
		fset:       fset,
		path:       packageDirectory,
		targetInfo: map[string]string{},
		linkNames:  map[string]string{},
		env:        env,
	}

	// Parse the program
	if err := prog.parse(); err != nil {
		return err
	}

	// Create the SSA
	allPackages, err := prog.buildPackages()
	if err != nil {
		return err
	}

	// Compute build order
	graph := NewGraph()
	for _, pkg := range allPackages {
		// Skip some packages as they are not real
		if pkg.Pkg.Path() == "unsafe" {
			continue
		}

		for _, imported := range pkg.Pkg.Imports() {
			// Skip some packages as they are not real
			if imported.Path() == "unsafe" {
				continue
			}
			graph.AddEdge(pkg, prog.ssaProg.Package(imported))
		}
	}

	// Create the buckets. These buckets represent packages that can be compiled in parallel
	buckets, err := graph.Buckets()
	if err != nil {
		return err
	}

	// Get the target
	target, err := compiler.NewTargetFromMap(prog.targetInfo)
	if err != nil {
		return err
	}

	// Initialize the target
	if err := target.Initialize(); err != nil {
		return err
	}

	// Create a compiler
	cc := compiler.NewCompiler(compiler.Options{
		Target:    target,
		LinkNames: prog.linkNames,
		Verbosity: compiler.Debug, // TODO: Create commandline option for this setting
	})

	// Compile the packages
	for _, bucket := range buckets {
		for _, pkg := range bucket {
			if err := cc.CompilePackage(context.Background(), pkg); err != nil {
				// TODO: Handle compiler errors more elegantly
				return err
			}
		}
	}

	// TODO: Generate the object file for this program
	// TODO: Compile the C standard library for the target (Probably Newlib)
	// TODO: Link the object file and C Standard Library
	// TODO: Convert the final binary image to the specified output binary type

	// Decide how to output
	switch filepath.Ext(output) {
	case ".bin":
		panic("Not implemented")
	case ".hex":
		panic("Not implemented")
	case ".ll":
		// Dump the module to the output file
		dumpModule(cc.Module(), output)
	}

	// Clean up the compiler
	cc.Dispose()

	// Clean up the target
	target.Dispose()

	return nil
}

func dumpModule(module llvm.LLVMModuleRef, fname string) error {
	// The path to the output must exist. Create it if it doesn't
	if stat, err := os.Stat(path.Dir(fname)); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(path.Dir(fname), 0750); err != nil {
			return err
		}
	} else if !stat.IsDir() {
		return os.ErrInvalid
	}

	// Finally, dump the module
	llvm.PrintModuleToFile(module, fname, nil)
	return nil
}
