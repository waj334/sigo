package builder

import (
	"context"
	"errors"
	"fmt"
	"go/types"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/tools/go/packages"

	"pkg.si-go.dev/go-mlir/mlir"
	goclang "pkg.si-go.dev/sigo/clang/binding/go"
	"pkg.si-go.dev/sigo/compiler/ssa"
	"pkg.si-go.dev/sigo/goir/binding/goir"
	"pkg.si-go.dev/sigo/llvm/tablegen"
)

type (
	optionsContextKey struct{}
)

func BuildPackages(ctx context.Context, options BuildOptions) error {
	// Add the options to the context
	ctx = context.WithValue(ctx, optionsContextKey{}, options)

	// Check output path with respect to the number of input packages
	if info, err := os.Stat(options.Output); err == nil && !info.IsDir() && len(options.Packages) > 1 {
		// Output must be a path if multiple packages were specified
		return ErrUnexpectedOutputPath
	}

	// Build each package
	for moduleRoot, pkgDir := range options.Packages {
		info, err := os.Stat(pkgDir)
		if err != nil {
			return errors.Join(ErrParserError, err)
		} else if info.IsDir() {
			return Build(ctx, moduleRoot, pkgDir)
		} else {
			// TODO: Allow a mix of package directories and individual .go files?
			panic("Not implemented")
		}
	}

	return nil
}

func Build(ctx context.Context, moduleDir, packageDir string) error {
	t := time.Now()
	defer func() {
		fmt.Printf("Build duration: %3fsec\n", time.Now().Sub(t).Seconds())
	}()

	// Get the options from the context
	options := ctx.Value(optionsContextKey{}).(BuildOptions)
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
	if err := StageGoRoot(goRootStaging, options.Environment); err != nil {
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

	// Perform an import analysis to locate the chip series TableGen file.
	importCfg := &packages.Config{
		Context: ctx,
		Dir:     moduleDir,
		Mode:    packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedModule,
		BuildFlags: []string{
			"-tags=" + options.Cpu,
		},
		Env: options.Environment.List(),
	}

	pkgs, err := packages.Load(importCfg, packageDir)
	if err != nil {
		return err
	}

	if len(pkgs) == 0 {
		return errors.New("there was a problem analyzing imports")
	}

	allPkgs := collectAllPackages(pkgs)

	// Find the platform TableGen file.
	platformFound := false
	tags := options.BuildTags
	alignment := int64(4)
	fpuEnabled := false

	var features []string
	var triplet string
	var archType string
	var cpuType string
	var fpuType string
	var machine mlir.LLVMTargetMachineRef

	for _, pkg := range allPkgs {
		fname := filepath.Join(pkg.Dir, "platform.td")
		_, err := os.Stat(fname)
		if os.IsNotExist(err) {
			continue
		}

		platformFound = true

		// Found the platform TableGen file. Parse it.
		rk := tablegen.NewRecordKeeper()

		// NOTE: The base TableGen definitions are next to go.mod in pkg.si-go.dev/chip.
		if !tablegen.ParseTableGenFile(fname, rk, []string{pkg.Module.Dir, filepath.Dir(fname)}) {
			return errors.New("failed to parse platform.td")
		}

		// Get the series def.
		allSeries := rk.GetDerivedRecords("Series")
		if len(allSeries) == 0 {
			return errors.New("no chip series defined")
		} else if len(allSeries) > 1 {
			return errors.New("multiple chip series encountered")
		}

		series := allSeries[0]
		seriesName := series.GetValueAsString("name")
		variants := series.GetValueAsListOfDefs("variants")

		// Find the variant matching the series.
		var variantTags []string
		variantExists := false
		for _, variant := range variants {
			variantName := strings.ToLower(variant.GetValueAsString("name"))
			if options.Cpu == variantName {
				variantExists = true
				variantTags = variant.GetValueAsListOfStrings("tags")
				break
			}
		}

		if !variantExists {
			return errors.New("this variant is not valid for the imported platform package")
		}

		// Get the architecture information.
		arch := series.GetValueAsDef("arch")
		archTags := arch.GetValueAsListOfStrings("tags")
		archFpu := arch.GetValueAsDef("fpu")

		archType = arch.GetValueAsString("arch")
		cpuType = arch.GetValueAsString("name")
		features = arch.GetValueAsListOfStrings("features")
		triplet = arch.GetValueAsString("triple")
		alignment = arch.GetValueAsInt("alignment")

		fpuType = archFpu.GetValueAsString("value")
		fpuFeatures := archFpu.GetValueAsListOfStrings("features")
		if fpuType == "none" {
			fpuFeatures = append(fpuFeatures, "soft-float")
		} else if options.Float == "hardfp" {
			fpuEnabled = true
			fpuFeatures = append(fpuFeatures, "fpregs")
			tags = append(tags, "fpu")
		} else {
			fpuFeatures = append(fpuFeatures, "soft-float")
		}

		features = append(features, fpuFeatures...)

		tags = append(tags, strings.ToLower(seriesName), strings.ToLower(options.Cpu), strings.ToLower(options.Float), strings.ToLower(archType))
		tags = append(tags, fpuFeatures...)
		tags = append(tags, variantTags...)
		tags = append(tags, archTags...)

		formattedFeatures := make([]string, len(features))
		for i, feature := range features {
			formattedFeatures[i] = "+" + feature
		}
		featureStr := strings.Join(formattedFeatures, ",")

		// Get the target from the triple
		target, err := mlir.NewTargetFromTriple(triplet)
		if err != nil {
			return errors.Join(ErrCodeGeneratorError, err)
		}

		machine = mlir.NewTargetMachine(
			target,
			triplet,
			cpuType,
			featureStr,
			mlir.LLVMCodeGenLevelNone,
			mlir.LLVMRelocDefault,
			mlir.LLVMCodeModelDefault)
		break
	}

	if !platformFound {
		return errors.New("no target platform could be determined")
	}

	targetLayout := mlir.NewTargetDataLayout(machine)

	// Set up sizes.
	sizes := types.StdSizes{
		WordSize: int64(targetLayout.PointerSize()),
		MaxAlign: alignment,
	}

	// Create a new program.
	program := ssa.NewProgram(&ssa.ProgramConfig{
		Tags: tags,
		//AdditionalPackages: additionalPackages,
		Environment: options.Environment.List(),
		PackagePath: packageDir,
		ModuleRoot:  moduleDir,
		GoRoot:      options.Environment.Value("GOROOT"),
		Sizes:       &sizes,
	})

	// Parse the package.
	fmt.Print("Parsing packages...")
	if err := program.Parse(ctx); err != nil {
		// TODO: Replace paths from the virtual GOROOT with the real paths in error strings.
		return err
	}
	fmt.Println("done")

	// Initialize MLIR.
	mlirCtx := mlir.NewContext()
	mlirCtx.RegisterAllLLVMTranslations()
	goir.DialectHandle().RegisterDialect(mlirCtx)
	goclang.RegisterDialects(mlirCtx)
	mlirCtx.LoadAllAvailableDialects()

	mlir.RegisterAllPasses()

	// Create the MLIR module.
	mlirModule := mlir.NewModule(mlir.NewUnknownLoc(mlirCtx))

	// Set module attributes before creating the SSA builder.
	goir.SetTargetDataLayout(mlirModule, targetLayout)
	goir.SetTargetTriple(mlirModule, triplet)

	// Create the SSA builder.
	builder := ssa.NewBuilder(ssa.Config{
		NumWorkers: options.NumJobs,
		Fset:       program.FileSet,
		Ctx:        mlirCtx,
		Sizes:      &sizes,
		Module:     mlirModule,
		Program:    program,
	})

	// Generate the SSA.
	fmt.Print("Building Go IR...")
	builder.GeneratePackages(ctx, program.OrderedPackages)
	fmt.Println("done")

	// Merge CIR modules from any import "C" preambles into the Go module.
	// Preambles were extracted during Parse via the CGo overlay pre-scan.
	if preamble := strings.Join(program.CGoPreambles, "\n"); preamble != "" {
		fmt.Print("Merging C preamble...")
		cirMod := goclang.LowerPreambleToMlir(preamble, triplet)
		if cirMod == nil {
			return errors.Join(ErrCodeGeneratorError, errors.New("CIR compilation of C preamble failed"))
		}
		if !cirMod.MergeInto(mlirCtx, mlirModule) {
			cirMod.Destroy()
			return errors.Join(ErrCodeGeneratorError, errors.New("failed to merge CIR module into Go module"))
		}
		cirMod.Destroy()
		fmt.Println("done")
	}

	// Create the output directory.
	outputDir := filepath.Dir(options.Output)
	if _, err := os.Stat(outputDir); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
			return err
		}
	}

	// Post IR generation:
	if options.DumpIR {
		filename := options.Output + ".dump.mlir"
		err := dumpMLIRModuleToFile(mlirModule, filename)
		if err != nil {
			return err
		}
	}

	// Verify the initial IR.
	if !mlirModule.Operation().Verify() {
		fmt.Fprintf(os.Stderr, "\n\nThe compiler produced invalid IR. The resulting binary may not be valid!\n"+
			"Please submit a bug ticket.\n\n")
		return ErrCodeGeneratorError
	}

	// Run the optimization passes
	fmt.Print("Optimizing Go IR...")
	if runOptimizerPass(mlirModule, options.DebugLowering).IsFailure() {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err, errors.New("optimization passes failed"))
	}
	fmt.Println("done")

	if options.DumpIR {
		filename := options.Output + ".dump.llvm.mlir"
		err := dumpMLIRModuleToFile(mlirModule, filename)
		if err != nil {
			return err
		}
	}

	// Generate the LLVM module
	llvmContext := mlir.NewLLVMContext()
	fmt.Print("Translating Go IR to LLVM IR...")
	llvmModule := mlir.TranslateModuleToLLVMIR(mlirModule.Operation(), llvmContext)

	if !options.GenerateDebugInfo {
		// Strip debug info
		llvmModule.StripModuleDebugInfo()
	}
	fmt.Println("done")

	// Add required constant globals to the LLVM module directly
	addConstantGlobals(llvmModule, options, fpuEnabled, targetLayout)

	if options.DumpIR {
		err := dumpModule(llvmModule, options.Output+".dump.ll")
		if err != nil {
			return err
		}
	}

	// Optimize modules
	fmt.Print("Optimizing LLVM IR...")
	if err = optimize(llvmModule, options.Optimization, machine); err != nil {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err)
	}
	fmt.Println("done")

	if options.DumpIR {
		err := dumpModule(llvmModule, options.Output+".dump.opt.ll")
		if err != nil {
			return err
		}
	}

	fmt.Print("Linking firmware image...")
	if err := link(linkOptions{
		triplet:       triplet,
		arch:          archType,
		cpu:           cpuType,
		fpu:           fpuType,
		floatEnabled:  fpuEnabled,
		features:      features,
		prog:          program,
		targetMachine: machine,
		module:        llvmModule,
	}, options,
	); err != nil {
		fmt.Println()
		return err
	}
	fmt.Println("done")

	// TODO: Clean up

	return nil
}

type linkOptions struct {
	triplet       string
	arch          string
	cpu           string
	fpu           string
	floatEnabled  bool
	features      []string
	prog          *ssa.Program
	targetMachine mlir.LLVMTargetMachineRef
	module        mlir.LLVMModuleRef
}

func link(options linkOptions, buildOptions BuildOptions) error {
	sigoRoot := buildOptions.Environment.Value("SIGOROOT")

	// Determine sysroot.
	abi := strings.Split(options.triplet, "-")[0]
	if options.floatEnabled {
		abi += "-fp"
	} else {
		abi += "-no-fp"
	}

	sysroot := filepath.Join(sigoRoot, "sysroots", abi)
	if _, err := os.Stat(sysroot); errors.Is(err, os.ErrNotExist) {
		return errors.Join(ErrCompilerFailed, fmt.Errorf("sysroot not found: %s", sysroot))
	}

	// Create the object file
	objectOut := filepath.Join(buildOptions.BuildDir, "firmware.o")
	if err := options.targetMachine.EmitToFile(
		options.module,
		objectOut,
		mlir.LLVMObjectFile,
	); err != nil {
		return errors.Join(ErrCodeGeneratorError, err)
	}

	// Get the toolchain.
	toolchain, err := findToolchain(buildOptions.Environment)
	if err != nil {
		return err
	}

	var artifacts []string
	if len(options.prog.LinkerScript) == 0 {
		return errors.New("no linker script found")
	}

	// Other arguments
	targetTriple := "--target=" + options.triplet
	elfOut := filepath.Join(buildOptions.BuildDir, "package.elf")
	args := []string{
		"--sysroot=" + sysroot,
		"--defsym=sigo_headGoroutine=runtime.headGoroutine",
		"--defsym=sigo_currentGoroutine=runtime.currentGoroutine",
		"--defsym=sigo_goroutineStackSize=runtime._goroutineStackSize",
		"-v",
		"--gc-sections",
		"-o", elfOut,
		"-nostdlib",
		"-L" + filepath.Join(sysroot, "lib"),
		"-L" + filepath.Join(sigoRoot, "runtime"),
		"-L" + filepath.Dir(options.prog.LinkerScript),
		"-T" + options.prog.LinkerScript,
		"-lc",
		"-lclang_rt.builtins",
	}

	if buildOptions.GenerateDebugInfo {
		args = append(args, "-g")
	}

	// Add all linker files
	for _, ld := range append(options.prog.Files[".ld"], options.prog.Files[".linker"]...) {
		args = append(args, "-L"+filepath.Dir(ld))
	}

	args = append(args, objectOut)
	args = append(args, artifacts...)

	// Compile all assembly files
	for _, asm := range append(options.prog.Files[".s"], options.prog.Files[".asm"]...) {
		// Format object file name
		fname, _ := filepath.EvalSymlinks(asm)
		objFile := filepath.Join(buildOptions.BuildDir, fmt.Sprintf("%s-%d.o", filepath.Base(asm), rand.Int()))

		assemblerArgs := []string{targetTriple,
			"-c", fname,
			func() string {
				if buildOptions.GenerateDebugInfo {
					return "-g"
				}
				return ""
			}(),
			func() string {
				if !options.floatEnabled {
					return "-mfloat-abi=softfp"
				}
				return "-mfloat-abi=hard"
			}(),
			"-o", objFile}

		// Append defines to the assembler arguments
		for def, val := range options.prog.Defines {
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
			return errors.Join(ErrCompilerFailed, err)
		}

		// Add this object file to the end of the linker command
		args = append(args, objFile)
	}

	// Invoke ld.lld to link the final binary image.
	lldCmd := exec.Command(toolchain.LD, args...)
	lldCmd.Stdout = nil
	lldCmd.Stderr = os.Stderr
	if err := lldCmd.Run(); err != nil {
		fmt.Println()
		fmt.Println("Command failed: ", lldCmd.String())
		return errors.Join(ErrCompilerFailed, err)
	}

	// Convert the final binary image to the specified output binary type
	switch filepath.Ext(buildOptions.Output) {
	case ".bin":
		objCopyCmd := exec.Command(toolchain.ObjCopy, "-O", "binary", elfOut, buildOptions.Output)
		if err := objCopyCmd.Run(); err != nil {
			output, _ := lldCmd.Output()
			return errors.Join(ErrCompilerFailed, err, errors.New(string(output)))
		}
	case ".hex":
		objCopyCmd := exec.Command(toolchain.ObjCopy, "-O", "ihex", elfOut, buildOptions.Output)
		if err := objCopyCmd.Run(); err != nil {
			output, _ := lldCmd.Output()
			return errors.Join(ErrCompilerFailed, err, errors.New(string(output)))
		}
	default:
		// Load the ELF into memory
		elfBytes, err := os.ReadFile(elfOut)
		if err != nil {
			return err
		}

		// Write the ELF as is to the output file
		if err = os.WriteFile(buildOptions.Output, elfBytes, 0644); err != nil {
			return err
		}
	}
	return nil
}

func dumpModule(module mlir.LLVMModuleRef, filename string) error {
	return os.WriteFile(filename, []byte(module.String()), 0644)
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

func optimize(module mlir.LLVMModuleRef, level string, machine mlir.LLVMTargetMachineRef) (err error) {
	var passes string

	// Create the pass builder options
	opts := mlir.NewPassBuilderOptions()
	defer opts.Dispose()

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
	err = mlir.LLVMRunPasses(module, passes, machine, opts)
	if err != nil {
		return err
	}

	// Verify the IR
	err = module.Verify(mlir.LLVMReturnStatusAction)
	if err != nil {
		return err
	}

	return nil
}

func addConstantGlobals(module mlir.LLVMModuleRef, options BuildOptions, floatEnabled bool, dataLayout mlir.LLVMTargetDataRef) {
	ctx := module.Context()
	intPtrType := ctx.IntPtrType(dataLayout)
	boolType := ctx.Int1Type()

	// Stack size for goroutines
	globalGoroutineStackSize := findOrCreateGlobal(module, intPtrType, "runtime._goroutineStackSize")
	alignment := dataLayout.PreferredAlignmentOfGlobal(globalGoroutineStackSize)
	constGoroutineStackSize := mlir.NewConstInt(intPtrType, uint64(align(uint(options.StackSize), alignment)), false)
	globalGoroutineStackSize.SetAlignment(alignment)
	globalGoroutineStackSize.SetInitializer(constGoroutineStackSize)
	globalGoroutineStackSize.SetLinkage(mlir.LLVMLinkageExternal)
	globalGoroutineStackSize.SetGlobalConstant(true)

	// FPU enable flag.
	globalFpuEnableFlag := findOrCreateGlobal(module, boolType, "runtime._fpuEnabled")
	alignment = dataLayout.PreferredAlignmentOfGlobal(globalFpuEnableFlag)

	var constFpuEnableFlag mlir.LLVMValueRef
	if floatEnabled {
		constFpuEnableFlag = mlir.NewConstInt(boolType, 1, false)
	} else {
		constFpuEnableFlag = mlir.NewConstInt(boolType, 0, false)
	}

	globalFpuEnableFlag.SetAlignment(alignment)
	globalFpuEnableFlag.SetInitializer(constFpuEnableFlag)
	globalFpuEnableFlag.SetLinkage(mlir.LLVMLinkageExternal)
	globalFpuEnableFlag.SetGlobalConstant(true)
}

func findOrCreateGlobal(module mlir.LLVMModuleRef, ty mlir.LLVMTypeRef, name string) mlir.LLVMValueRef {
	// Attempt to find the global value first
	for value := module.FirstGlobal(); !value.IsNull(); value = value.NextGlobal() {
		if value.Name() == name {
			if value.GlobalValueType() != ty {
				panic("global value type mismatch")
			}
			return value
		}
	}

	// Create the global value
	return module.AddGlobal(ty, name)
}

func align(n uint, m uint) uint {
	return n + (n % m)
}

func collectAllPackages(pkgs []*packages.Package) []*packages.Package {
	seen := make(map[*packages.Package]bool)
	var all []*packages.Package

	var visit func(pkg *packages.Package)
	visit = func(pkg *packages.Package) {
		if seen[pkg] {
			return
		}
		seen[pkg] = true
		all = append(all, pkg)
		for _, imp := range pkg.Imports {
			visit(imp)
		}
	}

	for _, pkg := range pkgs {
		visit(pkg)
	}
	return all
}

func dumpMLIRModuleToFile(module mlir.Module, filename string) error {
	_ = os.MkdirAll(filepath.Dir(filename), 0755)
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	if err != nil {
		return err
	}

	defer file.Close()

	flags := mlir.NewOpPrintingFlags().
		WithEnableDebugInfo(true, true).
		WithPrintNameLocAsPrefix()
	defer flags.Destroy()

	dumpStr := module.Operation().StringWithFlags(flags)
	_, err = file.WriteString(dumpStr)
	if err != nil {
		return err
	}
	return nil
}

// extractCGoPreambles collects the C preamble text from all import "C"
// declarations across the ordered package list. Multiple preambles are
// concatenated so they can be compiled together in a single CIR pass.
