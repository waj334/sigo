package builder

import (
	"context"
	"errors"
	"fmt"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/packages"
	"math/rand"
	"omibyte.io/sigo/compiler"
	"omibyte.io/sigo/llvm"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
)

type (
	optionsContextKey struct{}
)

func BuildPackages(ctx context.Context, options Options) error {
	// Add the options to the context
	ctx = context.WithValue(ctx, optionsContextKey{}, options)

	// Check output path with respect to the number of input packages
	if info, err := os.Stat(options.Output); err == nil && !info.IsDir() && len(options.Packages) > 1 {
		// Output must be a path if multiple packages were specified
		return ErrUnexpectedOutputPath
	}

	// Build each package
	for _, directory := range options.Packages {
		info, err := os.Stat(directory)
		if err != nil {
			return errors.Join(ErrParserError, err)
		} else if info.IsDir() {
			return Build(ctx, directory)
		} else {
			// TODO: Allow a mix of package directories and individual .go files?
			panic("Not implemented")
		}
	}

	return nil
}

func Build(ctx context.Context, packageDir string) error {
	// Get the options from the context
	options := ctx.Value(optionsContextKey{}).(Options)

	fset := token.NewFileSet()
	config := types.Config{}

	compilerOptions := compiler.NewOptions()
	compilerOptions.Verbosity = options.CompilerVerbosity
	compilerOptions.GenerateDebugInfo = options.GenerateDebugInfo

	// Create the build directory
	if len(options.BuildDir) == 0 {
		// Create a random build directory
		options.BuildDir = filepath.Join(options.Environment.Value("SIGOCACHE"), fmt.Sprintf("sigo-build-%d", rand.Int()))
		if err := os.MkdirAll(options.BuildDir, os.ModeDir); err != nil {
			return err
		}
		// Delete the build directory when done
		defer os.RemoveAll(options.BuildDir)
	} else {
		stat, err := os.Stat(options.BuildDir)
		if os.IsNotExist(err) {
			if err = os.MkdirAll(options.BuildDir, os.ModeDir); err != nil {
				return err
			}
		} else if !stat.IsDir() {
			return errors.Join(ErrUnexpectedOutputPath, errors.New("specified build directory is not a directory"))
		}
	}

	// Create the temporary directory required by the package loader
	if _, err := os.Stat(options.Environment.Value("GOTMPDIR")); os.IsNotExist(err) {
		if err = os.MkdirAll(options.Environment.Value("GOTMPDIR"), os.ModeDir); err != nil {
			panic(err)
		}
	}

	// Create the staging directory for GOROOT
	goRootStaging := filepath.Join(options.Environment.Value("GOTMPDIR"), fmt.Sprintf("goroot-%d", rand.Int()))
	if err := os.MkdirAll(goRootStaging, os.ModeDir); err != nil {
		panic(err)
	}

	// Stage the GOROOT that will be used for parsing the packages.
	// NOTE: GOROOT must be unmodified at this point so that directories can be symlinked accurately.
	if err := stageGoRoot(goRootStaging, options.Environment); err != nil {
		panic(err)
	}

	// Delete the staging directory when the build is done.
	defer os.RemoveAll(goRootStaging)

	// Now modify the GOROOT value in the environment
	options.Environment["GOROOT"] = goRootStaging

	// Create a path mapping from the staged goroot to sigoroot
	// so file paths are correct in any debug info generated
	// later.
	compilerOptions.PathMappings[goRootStaging] = options.Environment.Value("SIGOROOT")

	// Create a new program
	prog := Program{
		config:     config,
		fset:       fset,
		path:       packageDir,
		targetInfo: map[string]string{},
		pkgs:       []*packages.Package{},
		env:        options.Environment,
		options:    compilerOptions,
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
	if err = target.Initialize(); err != nil {
		return err
	}

	compilerOptions.Target = target

	// Create a compiler
	cc, compilerCtx := compiler.NewCompiler(*compilerOptions)

	runtimePackages := []string{
		"runtime/internal/allocator",
		"runtime",
	}

	for _, runtimePackage := range runtimePackages {
		// Get the runtime package that all packages will have an implicit dependency on
		runtimePkg := prog.ssaProg.ImportedPackage(runtimePackage)
		if runtimePkg == nil {
			panic("missing \"" + runtimePackage + "\" package")
		}

		// Compile the runtime package first
		if err = cc.CompilePackage(ctx, compilerCtx, runtimePkg); err != nil {
			// TODO: Handle compiler errors more elegantly
			return err
		}
	}

	// Compile the packages
	for _, bucket := range buckets {
		for _, pkg := range bucket {
			// Skip previously compiled runtime packages
			skip := false
			for _, runtimePackage := range runtimePackages {
				if pkg.Pkg.Path() == runtimePackage {
					// Do not build this package more than once
					skip = true
					break
				}
			}

			if skip {
				continue
			}

			if err = cc.CompilePackage(ctx, compilerCtx, pkg); err != nil {
				// TODO: Handle compiler errors more elegantly
				return err
			}
		}
	}

	// Finalize the compiler
	cc.Finalize()

	dumpOut := options.Output
	if filepath.Ext(dumpOut) != ".ll" {
		dumpOut += ".dump.ll"
	}
	dumpModule(cc.Module(), dumpOut)

	// Verfiy the IR
	if ok, errMsg := llvm.VerifyModule2(cc.Module(), llvm.LLVMVerifierFailureAction(llvm.ReturnStatusAction)); !ok {
		if options.DumpOnVerifyError {
			if filepath.Ext(options.Output) != ".ll" {
				options.Output += ".dump.ll"
			}
			dumpModule(cc.Module(), options.Output)
		}
		return errors.Join(ErrCodeGeneratorError, errors.New(errMsg))
	}

	// Generate the object file for this program
	objectOut := filepath.Join(options.BuildDir, "package.o")
	if ok, errMsg := llvm.TargetMachineEmitToFile2(target.Machine(), cc.Module(), objectOut, llvm.LLVMCodeGenFileType(llvm.ObjectFile)); !ok {
		return errors.Join(ErrCodeGeneratorError, errors.New(errMsg))
	}

	// TODO: Select the proper build of picolibc
	libCDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/picolibc", target.Triple(), "lib")

	// Select build of runtime-rt
	libCompilerRTDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/compiler-rt/lib", target.Triple())

	// Locate clang
	executablePostfix := ""
	if runtime.GOOS == "windows" {
		executablePostfix = ".exe"
	}
	clangExecutable := filepath.Join(options.Environment.Value("SIGOROOT"), "bin/clang"+executablePostfix)
	if _, err = os.Stat(clangExecutable); os.IsNotExist(err) {
		// Try build folder if this is a dev build
		clangExecutable = filepath.Join(options.Environment.Value("SIGOROOT"), "build/llvm-build/bin/clang"+executablePostfix)
		if _, err = os.Stat(clangExecutable); os.IsNotExist(err) {
			return errors.New("could not locate clang in SIGOROOT")
		}
	}

	// Other arguments
	targetTriple := "--target=" + target.Triple()
	elfOut := filepath.Join(options.BuildDir, "package.elf")
	args := []string{
		"-v",
		targetTriple,
		"-fuse-ld=lld",
		"-o", elfOut,
		"-L" + libCDir,
		"-L" + libCompilerRTDir,
		"-W1,-L" + filepath.Join(options.Environment.Value("SIGOROOT"), "runtime"),
		"-T" + prog.targetLinkerFile,
		objectOut,
	}

	// Compile all assembly files
	for _, asm := range prog.assemblyFiles {
		// Format object file name
		objFile := filepath.Join(options.BuildDir, fmt.Sprintf("%s-%d.o", filepath.Base(asm), rand.Int()))

		// Invoke Clang to compile the final binary
		clangCmd := exec.Command(clangExecutable, targetTriple, "-c", asm, "-o", objFile)
		clangCmd.Stdout = os.Stdout
		clangCmd.Stderr = os.Stderr
		if err = clangCmd.Run(); err != nil {
			return errors.Join(ErrClangFailed, err)
		}

		// Add this object file to the end of the linker command
		args = append(args, objFile)
	}

	// Invoke Clang to compile the final binary
	clangCmd := exec.Command(clangExecutable, args...)
	clangCmd.Stdout = os.Stdout
	clangCmd.Stderr = os.Stderr
	if err = clangCmd.Run(); err != nil {
		return errors.Join(ErrClangFailed, err)
	}

	// Convert the final binary image to the specified output binary type
	switch filepath.Ext(options.Output) {
	case ".bin":
		objCopyCmd := exec.Command(filepath.Join(filepath.Dir(clangExecutable), "llvm-objcopy"), "-O", "binary", elfOut, options.Output)
		if err = objCopyCmd.Run(); err != nil {
			output, _ := clangCmd.Output()
			return errors.Join(ErrClangFailed, err, errors.New(string(output)))
		}
	case ".hex":
		objCopyCmd := exec.Command(filepath.Join(filepath.Dir(clangExecutable), "llvm-objcopy"), "-O", "ihex", elfOut, options.Output)
		if err = objCopyCmd.Run(); err != nil {
			output, _ := clangCmd.Output()
			return errors.Join(ErrClangFailed, err, errors.New(string(output)))
		}
	case ".ll":
		// Dump the module to the output file
		dumpModule(cc.Module(), options.Output)
	default:
		// Load the ELF into memory
		elfBytes, err := os.ReadFile(elfOut)
		if err != nil {
			return err
		}

		// Write the ELF as is to the output file
		if err = os.WriteFile(options.Output, elfBytes, 0644); err != nil {
			return err
		}
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

func stageGoRoot(stageDir string, env Env) error {
	// Symlink the "pkg" directory from GOROOT into the staging directory
	goPkgPath := path.Join(env.Value("GOROOT"), "pkg")
	stagedPkgPath := path.Join(stageDir, "pkg")
	if err := os.Symlink(goPkgPath, stagedPkgPath); err != nil {
		return err
	}

	// Symlink the standard library from SiGo into the staging directory
	sigoStlPath := path.Join(env.Value("SIGOROOT"), "src")
	stagedStlPath := path.Join(stageDir, "src")
	if err := os.Symlink(sigoStlPath, stagedStlPath); err != nil {
		return err
	}

	return nil
}
