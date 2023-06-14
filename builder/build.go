package builder

import (
	"context"
	"errors"
	"fmt"
	"go/token"
	"go/types"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
	"hash/crc32"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"

	"omibyte.io/sigo/compiler"
	"omibyte.io/sigo/llvm"
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
	compilerOptions.PrimitivesAsCTypes = options.CTypeNames

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

	// TODO: Detect the target architecture by some other means
	options.Environment["GOARCH"] = "arm"

	// Create a new program
	prog := Program{
		config:        config,
		fset:          fset,
		path:          packageDir,
		targetInfo:    map[string]string{},
		pkgs:          []*packages.Package{},
		env:           options.Environment,
		options:       compilerOptions,
		assemblyFiles: map[string]struct{}{},
		linkerFiles:   map[string]struct{}{},
	}

	// Parse the program
	if err := prog.parse(ctx); err != nil {
		return err
	}

	// Create the SSA
	allPackages, err := prog.buildPackages()
	if err != nil {
		return err
	}

	// TODO: Check target triple now to fail early!!!
	/*
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
	*/

	stopChan := make(chan struct{})
	errChan := make(chan error)
	go func() {
		select {
		case <-stopChan:
			return
		case err := <-errChan:
			panic(err)
		}
	}()

	numJobs := options.NumJobs
	if numJobs == 0 {
		numJobs = 1
	}

	// This will hold the compiled modules
	compilers := map[string]*compiler.Compiler{}
	var mu sync.Mutex

	// Create a bounded worker pool
	var wg sync.WaitGroup
	wg.Add(numJobs)

	// Create the channel that packages to be compiled will be signaled on
	pkgChan := make(chan *ssa.Package, numJobs)

	// Create the compiler pool
	for i := 0; i < numJobs; i++ {
		go func() {
			defer wg.Done()
			for pkg := range pkgChan {
				// Get the target
				target, _ := compiler.NewTargetFromMap(prog.targetInfo)

				// Initialize the target
				target.Initialize()

				opts := compilerOptions.WithTarget(target)

				// Create a compiler
				cc, compilerCtx := compiler.NewCompiler(pkg.Pkg.Path(), opts)

				// Compile the package
				if err = cc.CompilePackage(ctx, compilerCtx, pkg); err != nil {
					// TODO: Handle compiler errors more elegantly
					errChan <- err
					return
				}

				// Finish up
				cc.Finalize()

				// Store the module
				mu.Lock()
				compilers[pkg.Pkg.Path()] = cc
				mu.Unlock()
			}
		}()
	}

	// Start signaling packages to be compiled by the pool
	for _, pkg := range allPackages {
		pkgChan <- pkg
	}
	close(pkgChan)

	// Wait for remaining jobs to complete
	wg.Wait()

	// Create the init module last
	{
		// Create lookup table for packages
		lookup := map[*types.Package]int64{}
		for i, pkg := range allPackages {
			lookup[pkg.Pkg] = int64(i)
		}

		// Create a directed graph so that the order in which the init functions should be call can be determined.
		graph := simple.NewDirectedGraph()
		for _, pkg := range allPackages {
			from, _ := graph.NodeWithID(lookup[pkg.Pkg])
			for _, imported := range pkg.Pkg.Imports() {
				to, _ := graph.NodeWithID(lookup[imported])
				e := graph.NewEdge(from, to)
				graph.SetEdge(e)
			}
		}

		// Perform a topological sort
		result, err := topo.Sort(graph)
		if err != nil {
			return err
		}

		var pkgs []*ssa.Package
		for _, node := range result {
			// Prepend to get the correct call order
			pkgs = append([]*ssa.Package{allPackages[node.ID()]}, pkgs...)
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

		cc, llctx := compiler.NewCompiler("init", compilerOptions.WithTarget(target))
		cc.CreateInitLib(llctx, pkgs)
		cc.Finalize()
		compilers["__init__"] = cc
	}

	var objFiles []string

	// Process each module
	moduleChan := make(chan struct {
		pkgPath string
		cc      *compiler.Compiler
	}, numJobs)
	wg.Add(numJobs)

	for i := 0; i < numJobs; i++ {
		go func() {
			defer wg.Done()
			for job := range moduleChan {
				dumpOut := options.Output + "." + strings.ReplaceAll(job.pkgPath, "/", "_") + ".dump.ll"

				// Verfiy the IR
				verifySucceeded, errMsg := llvm.VerifyModule2(job.cc.Module(), llvm.LLVMVerifierFailureAction(llvm.ReturnStatusAction))
				if (!verifySucceeded && options.DumpOnVerifyError) || options.DumpIR {
					dumpModule(job.cc.Module(), dumpOut)
				}

				if !verifySucceeded {
					errChan <- errors.Join(ErrCodeGeneratorError, errors.New(job.pkgPath+":\n"+errMsg))
				}

				nameHash := crc32.Checksum([]byte(job.pkgPath), crc32.IEEETable)
				objectOut := filepath.Join(options.BuildDir, strconv.Itoa(int(nameHash))+".o")

				mu.Lock()
				objFiles = append(objFiles, objectOut)
				mu.Unlock()

				// Generate the object file for this program
				if ok, errMsg := llvm.TargetMachineEmitToFile2(job.cc.Options().Target.Machine(), job.cc.Module(), objectOut, llvm.LLVMCodeGenFileType(llvm.ObjectFile)); !ok {
					errChan <- errors.Join(ErrCodeGeneratorError, errors.New(job.pkgPath+":\n"+errMsg))
				}
			}
		}()
	}

	for pkgPath, cc := range compilers {
		moduleChan <- struct {
			pkgPath string
			cc      *compiler.Compiler
		}{pkgPath: pkgPath, cc: cc}
	}
	close(moduleChan)
	wg.Wait()

	// Done with error monitor
	stopChan <- struct{}{}
	close(stopChan)
	close(errChan)

	triple := prog.targetInfo["triple"]

	// TODO: Select the proper build of picolibc
	libCDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/picolibc", triple, "lib")

	// Select build of runtime-rt
	libCompilerRTDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/compiler-rt/lib", triple)

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
	targetTriple := "--target=" + triple
	elfOut := filepath.Join(options.BuildDir, "package.elf")
	args := []string{
		"-v",
		targetTriple,
		"-fuse-ld=lld",
		"-W1,--gc-sections",
		"-o", elfOut,
		"-L" + libCDir,
		"-L" + libCompilerRTDir,
		"-W1,-L" + filepath.Join(options.Environment.Value("SIGOROOT"), "runtime"),
		"-L" + filepath.Dir(prog.targetLinkerFile),
		"-T" + prog.targetLinkerFile,
	}

	// Add all linker files
	for ld, _ := range prog.linkerFiles {
		args = append(args, "-L"+filepath.Dir(ld))
	}

	args = append(args, objFiles...)

	// Compile all assembly files
	for asm, _ := range prog.assemblyFiles {
		// Format object file name
		objFile := filepath.Join(options.BuildDir, fmt.Sprintf("%s-%d.o", filepath.Base(asm), rand.Int()))

		assemblerArgs := []string{targetTriple,
			"-c", asm,
			"-ffunction-sections",
			"-fdata-sections",
			"-o", objFile}

		// Append defines to the assembler arguments
		for def, val := range prog.defines {
			if len(val) == 0 {
				assemblerArgs = append(assemblerArgs,
					"-D"+def)
			} else {
				assemblerArgs = append(assemblerArgs,
					fmt.Sprintf("-D%s=%s", def, val))
			}
		}

		// Invoke Clang to compile the assembly sources
		clangCmd := exec.Command(clangExecutable, assemblerArgs...)

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

	// Clean up the compilers
	for _, cc := range compilers {
		cc.Dispose()
	}

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

func symbolName(pkg *types.Package, name string) string {
	path := "_"
	// pkg is nil for objects in Universe scope and possibly types
	// introduced via Eval (see also comment in object.sameId)
	if pkg != nil && pkg.Path() != "" {
		path = pkg.Path()
	}
	return path + "." + name
}
