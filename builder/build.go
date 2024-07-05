package builder

import (
	"context"
	"errors"
	"fmt"
	"go/types"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"

	"omibyte.io/sigo/compiler/ssa"
	"omibyte.io/sigo/llvm"
	"omibyte.io/sigo/mlir"
	"omibyte.io/sigo/targets"
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
	t := time.Now()
	defer func() {
		fmt.Printf("Build duration: %3fsec\n", time.Now().Sub(t).Seconds())
	}()

	// Get the options from the context
	options := ctx.Value(optionsContextKey{}).(Options)
	pathMappings := map[string]string{}

	// Create the build directory
	if len(options.BuildDir) == 0 {
		// Create a random build directory
		options.BuildDir = filepath.Join(options.Environment.Value("SIGOCACHE"), fmt.Sprintf("sigo-build-%d", rand.Int()))
		if err := os.MkdirAll(options.BuildDir, os.ModePerm); err != nil {
			return err
		}
		// Delete the build directory when done
		if !options.KeepWorkDir {
			defer os.RemoveAll(options.BuildDir)
		} else {
			fmt.Printf("Work directory: %s\n", options.BuildDir)
		}
	} else {
		stat, err := os.Stat(options.BuildDir)
		if os.IsNotExist(err) {
			if err = os.MkdirAll(options.BuildDir, os.ModePerm); err != nil {
				return err
			}
		} else if !stat.IsDir() {
			return errors.Join(ErrUnexpectedOutputPath, errors.New("specified build directory is not a directory"))
		}
	}

	// Create the temporary directory required by the package loader
	if _, err := os.Stat(options.Environment.Value("GOTMPDIR")); os.IsNotExist(err) {
		if err = os.MkdirAll(options.Environment.Value("GOTMPDIR"), os.ModePerm); err != nil {
			panic(err)
		}
	}

	// Create the staging directory for GOROOT
	goRootStaging := filepath.Join(options.Environment.Value("GOTMPDIR"), fmt.Sprintf("goroot-%d", rand.Int()))
	if err := os.MkdirAll(goRootStaging, os.ModePerm); err != nil {
		panic(err)
	}

	// Stage the GOROOT that will be used for parsing the packages.
	// NOTE: GOROOT must be unmodified at this point so that directories can be symlinked accurately.
	if err := stageGoRoot(goRootStaging, options.Environment); err != nil {
		panic(err)
	}

	// Delete the staging directory when the build is done.
	if !options.KeepWorkDir {
		defer os.RemoveAll(goRootStaging)
	} else {
		fmt.Printf("GOROOT staging directory: %s\n", options.BuildDir)
	}

	// Now modify the GOROOT value in the environment
	options.Environment["GOROOT"] = goRootStaging

	// Create a path mapping from the staged goroot to sigoroot
	// so file paths are correct in any debug info generated
	// later.
	pathMappings[goRootStaging] = options.Environment.Value("SIGOROOT")

	// Get the target information
	var tags = options.BuildTags
	var additionalPackages []string
	targetInfo, err := targets.All().FindByChip(options.Cpu)
	if err != nil {
		// Fallback to series
		targetInfo, err = targets.All().FindBySeries(options.Cpu)
		if err != nil {
			return err
		}
	} else {
		tags = append(tags, options.Cpu, targetInfo.Architecture, targetInfo.Series, options.Float)
		tags = append(tags, targetInfo.Tags...)
		additionalPackages = append(additionalPackages, strings.ReplaceAll(targetInfo.ChipPackage, "${chip}", options.Cpu))
	}

	// TODO: Detect the target architecture by some other means
	options.Environment["GOARCH"] = "arm"
	arch := strings.Split(targetInfo.Triple, "-")[0]
	float := "nofp"
	switch targetInfo.Float {
	case "hardfp":
		if options.Float == "softfp" {
			float = "nofp"
			targetInfo.Features = append(targetInfo.Features, "soft-float")
		} else {
			float = "fp"
			/// TODO: There are different FP instruction features available for different chips. Find a better way of
			/// specifying those respective features.
		}
	default:
		float = "nofp"
		targetInfo.Features = append(targetInfo.Features, "soft-float")
	}

	// Create the machine target
	target, err := targetInfo.CreateTarget()
	if err != nil {
		return err
	}
	targetMachine := targetInfo.CreateTargetMachine(target)
	targetLayout := llvm.CreateTargetDataLayout(targetMachine)

	// Set up sizes.
	sizes := types.StdSizes{
		WordSize: int64(llvm.PointerSize(targetLayout)),
		MaxAlign: int64(targetInfo.Alignment),
	}

	// Create a new program.
	program := ssa.NewProgram(&ssa.ProgramConfig{
		Tags:               tags,
		AdditionalPackages: additionalPackages,
		Environment:        options.Environment.List(),
		PackagePath:        packageDir,
		GoRoot:             options.Environment.Value("GOROOT"),
		Sizes:              &sizes,
	})

	// Parse the package.
	fmt.Print("Parsing packages...")
	if err := program.Parse(ctx); err != nil {
		// TODO: Replace paths from the virtual GOROOT with the real paths in error strings.
		return err
	}
	fmt.Println("done")

	// Initialize MLIR.
	mlirCtx := mlir.ContextCreate()
	mlir.DialectHandleRegisterDialect(mlir.GetDialectHandle__go__(), mlirCtx)
	mlir.ContextLoadAllAvailableDialects(mlirCtx)

	// Create the MLIR module.
	mlirModule := mlir.ModuleCreateEmpty(mlir.LocationUnknownGet(mlirCtx))

	// Create the SSA builder.
	builder := ssa.NewBuilder(ssa.Config{
		NumWorkers: options.NumJobs,
		Fset:       program.FileSet,
		Ctx:        mlirCtx,
		Sizes:      &sizes,
		Module:     mlirModule,
		Program:    program,
	})

	// Set module attributes
	dataLayout := llvm.CreateTargetDataLayout(targetMachine)
	mlir.GoSetTargetDataLayout(mlirModule, dataLayout)
	mlir.GoSetTargetTriple(mlirModule, targetInfo.Triple)

	// Generate the SSA.
	fmt.Print("Building Go IR...")
	builder.GeneratePackages(ctx, program.OrderedPackages)
	fmt.Println("done")

	// dump the module to a string
	fname, _ := filepath.Abs(options.Output + ".dump.mlir")

	// Post IR generation:
	if options.DumpIR {

		// The path to the output must exist. Create it if it doesn't
		if stat, err := os.Stat(path.Dir(fname)); errors.Is(err, os.ErrNotExist) {
			if err := os.MkdirAll(path.Dir(fname), 0750); err != nil {
				return err
			}
		} else if !stat.IsDir() {
			return os.ErrInvalid
		}

		mlir.ModuleDumpToFile(mlirModule, fname)
	}

	// Run the optimization passes
	passDumpDir, _ := filepath.Abs(options.Output)
	passDumpDir = filepath.Dir(passDumpDir)
	passDumpName := filepath.Base(options.Output)
	fmt.Print("Optimizing Go IR...")
	// TODO: add switch for debug mode.
	if mlir.LogicalResultIsFailure(mlir.GoOptimizeModule(mlirModule, passDumpName, passDumpDir, false)) {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err, errors.New("optimization passes failed"))
	}
	fmt.Println("done")

	if options.DumpIR {
		// dump the module to a string
		fname := options.Output + ".dump.llvm.mlir"

		// The path to the output must exist. Create it if it doesn't
		if stat, err := os.Stat(path.Dir(fname)); errors.Is(err, os.ErrNotExist) {
			if err := os.MkdirAll(path.Dir(fname), 0750); err != nil {
				return err
			}
		} else if !stat.IsDir() {
			return os.ErrInvalid
		}

		mlir.ModuleDumpToFile(mlirModule, fname)
	}

	// Initialize the LLVMIR translator
	mlir.InitModuleTranslation(mlir.ModuleGetContext(mlirModule))

	// Generate the LLVM module
	llvmContext := llvm.ContextCreate()
	fmt.Print("Translating Go IR to LLVM IR...")
	llvmModule := mlir.TranslateModuleToLLVMIR(mlirModule, llvmContext, "module")
	if !options.GenerateDebugInfo {
		// Strip debug info
		llvm.StripModuleDebugInfo(llvmModule)
	}
	fmt.Println("done")

	// Add required constant globals to the LLVM module directly
	addConstantGlobals(llvmModule, options, dataLayout)

	if options.DumpIR {
		dumpModule(llvmModule, options.Output+".dump.ll")
	}

	// Optimize modules
	fmt.Print("Optimizing LLVM IR...")
	if err = optimize(llvmModule, options.Optimization, targetMachine); err != nil {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err)
	}
	fmt.Println("done")

	if options.DumpIR {
		dumpModule(llvmModule, options.Output+".dump.opt.ll")
	}

	fmt.Print("Linking firmware image...")
	if err := link(options, targetInfo, arch, float, program, targetMachine, llvmModule); err != nil {
		fmt.Println()
		return err
	}
	fmt.Println("done")

	//TODO: Clean up

	return nil
}

func link(options Options, targetInfo targets.TargetInfo, arch string, float string, prog *ssa.Program, targetMachine llvm.LLVMTargetMachineRef, module llvm.LLVMModuleRef) error {
	// Create the object file
	objectOut := filepath.Join(options.BuildDir, "firmware.o")
	if ok, errMsg := llvm.TargetMachineEmitToFile2(targetMachine, module, objectOut, llvm.LLVMCodeGenFileType(llvm.ObjectFile)); !ok {
		return errors.Join(ErrCodeGeneratorError, errors.New(errMsg))
	}

	// TODO: Select the proper build of picolibc
	libCDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/picolibc", targetInfo.Triple, arch+"+"+float, "lib")

	// Select build of runtime-rt
	libCompilerRTDir := filepath.Join(options.Environment.Value("SIGOROOT"), "lib/compiler-rt", targetInfo.Triple, arch+"+"+float, "lib", targetInfo.Triple)

	// Get the toolchain.
	toolchain, err := findToolchain(options.Environment)
	if err != nil {
		return err
	}

	// Other arguments
	targetTriple := "--target=" + targetInfo.Triple
	elfOut := filepath.Join(options.BuildDir, "package.elf")
	args := []string{
		"-v",
		"--gc-sections",
		"-o", elfOut,
		"-nostdlib",
		"-L" + libCDir,
		"-L" + libCompilerRTDir,
		"-L" + filepath.Join(options.Environment.Value("SIGOROOT"), "runtime"),
		"-L" + filepath.Dir(prog.LinkerScript),
		"-T" + prog.LinkerScript,
		"-lc",
		"-lclang_rt.builtins-" + arch,
	}

	if options.GenerateDebugInfo {
		args = append(args, "-g")
	}

	// Add all linker files
	for _, ld := range append(prog.Files[".ld"], prog.Files[".linker"]...) {
		args = append(args, "-L"+filepath.Dir(ld))
	}

	args = append(args, objectOut)

	// Compile all assembly files
	for _, asm := range append(prog.Files[".s"], prog.Files[".asm"]...) {
		// Format object file name
		objFile := filepath.Join(options.BuildDir, fmt.Sprintf("%s-%d.o", filepath.Base(asm), rand.Int()))

		assemblerArgs := []string{targetTriple,
			"-c", asm,
			func() string {
				if options.GenerateDebugInfo {
					return "-g"
				}
				return ""
			}(),
			"-o", objFile}

		// Append defines to the assembler arguments
		for def, val := range prog.Defines {
			if len(val) == 0 {
				assemblerArgs = append(assemblerArgs,
					"-D"+def)
			} else {
				assemblerArgs = append(assemblerArgs,
					fmt.Sprintf("-D%s=%s", def, val))
			}
		}

		// Invoke Clang to compile the assembly sources
		clangCmd := exec.Command(toolchain.CC, assemblerArgs...)

		clangCmd.Stdout = nil
		clangCmd.Stderr = os.Stderr
		if err := clangCmd.Run(); err != nil {
			fmt.Println()
			fmt.Println("Command failed: ", clangCmd.String())
			return errors.Join(ErrClangFailed, err)
		}

		// Add this object file to the end of the linker command
		args = append(args, objFile)
	}

	// Invoke ld.lld to compile the final binary
	lldCmd := exec.Command(toolchain.LD, args...)
	lldCmd.Stdout = nil
	lldCmd.Stderr = os.Stderr
	if err := lldCmd.Run(); err != nil {
		fmt.Println()
		fmt.Println("Command failed: ", lldCmd.String())
		return errors.Join(ErrClangFailed, err)
	}

	// Convert the final binary image to the specified output binary type
	switch filepath.Ext(options.Output) {
	case ".bin":
		objCopyCmd := exec.Command(toolchain.ObjCopy, "-O", "binary", elfOut, options.Output)
		if err := objCopyCmd.Run(); err != nil {
			output, _ := lldCmd.Output()
			return errors.Join(ErrClangFailed, err, errors.New(string(output)))
		}
	case ".hex":
		objCopyCmd := exec.Command(toolchain.ObjCopy, "-O", "ihex", elfOut, options.Output)
		if err := objCopyCmd.Run(); err != nil {
			output, _ := lldCmd.Output()
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

func optimize(module llvm.LLVMModuleRef, level string, machine llvm.LLVMTargetMachineRef) (err error) {
	var passes string

	// Create the pass builder options
	opts := llvm.CreatePassBuilderOptions()
	defer llvm.DisposePassBuilderOptions(opts)

	// Match Clang's optimization settings
	switch level {
	case "1":
		passes = "default<O1>"
	case "2":
		passes = "default<O2>"
	case "3":
		passes = "default<O3>"
	case "s":
		passes = "default<O0>"
	case "z":
		passes = "default<Oz>"
	case "d":
		return nil
	default:
		passes = "default<O0>"
	}

	// Run the passes
	llvm.RunPasses(module, passes, machine, opts)

	// Verfiy the IR
	if ok, errMsg := llvm.VerifyModule2(module, llvm.LLVMVerifierFailureAction(llvm.ReturnStatusAction)); !ok {
		return errors.New(errMsg)
	}

	return
}

func addConstantGlobals(module llvm.LLVMModuleRef, options Options, dataLayout llvm.LLVMTargetDataRef) {
	intPtrType := llvm.IntPtrTypeInContext(llvm.GetModuleContext(module), dataLayout)

	// Stack size for goroutines
	globalGoroutineStackSize := findOrCreateGlobal(module, intPtrType, "runtime._goroutineStackSize")
	alignment := llvm.PreferredAlignmentOfGlobal(dataLayout, globalGoroutineStackSize)
	constGoroutineStackSize := llvm.ConstInt(intPtrType, uint64(align(uint(options.StackSize), alignment)), false)
	llvm.SetAlignment(globalGoroutineStackSize, alignment)
	llvm.SetInitializer(globalGoroutineStackSize, constGoroutineStackSize)
	llvm.SetLinkage(globalGoroutineStackSize, llvm.ExternalLinkage)
	llvm.SetGlobalConstant(globalGoroutineStackSize, true)
}

func findOrCreateGlobal(module llvm.LLVMModuleRef, ty llvm.LLVMTypeRef, name string) llvm.LLVMValueRef {
	// Attempt to find the global value first
	for value := llvm.GetFirstGlobal(module); !value.IsNil(); value = llvm.GetNextGlobal(value) {
		if llvm.GetValueName2(value) == name {
			if !llvm.TypeIsEqual(llvm.GlobalGetValueType(value), ty) {
				panic("global value type mismatch")
			}
			return value
		}
	}

	// Create the global value
	return llvm.AddGlobal(module, ty, name)
}

func align(n uint, m uint) uint {
	return n + (n % m)
}
