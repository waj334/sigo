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
	tags := append([]string{"baremetal"}, options.BuildTags...)
	pointerAlignment := int64(4)
	stackAlignment := int64(8)
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
		pointerAlignment = arch.GetValueAsInt("pointerAlignment")
		stackAlignment = arch.GetValueAsInt("stackAlignment")

		fpuType = archFpu.GetValueAsString("value")
		fpuFeatures := archFpu.GetValueAsListOfStrings("features")
		if fpuType == "nofpu" {
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

		// Set GOARCH and GOOS so that go/packages.Load() correctly
		// evaluates file-name suffixes and //go:build constraints
		// for the target architecture.
		options.Environment["GOARCH"] = goarch(archType)
		options.Environment["GOOS"] = "linux"

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

		// Map user optimization level to LLVM CodeGen optimization level.
		var codeGenLevel mlir.LLVMCodeGenOptLevel
		switch options.Optimization {
		case "1":
			codeGenLevel = mlir.LLVMCodeGenLevelLess
		case "2", "s":
			codeGenLevel = mlir.LLVMCodeGenLevelDefault
		case "3":
			codeGenLevel = mlir.LLVMCodeGenLevelAggressive
		case "z":
			codeGenLevel = mlir.LLVMCodeGenLevelDefault
		default:
			codeGenLevel = mlir.LLVMCodeGenLevelNone
		}

		machine = mlir.NewTargetMachine(
			target,
			triplet,
			cpuType,
			featureStr,
			codeGenLevel,
			mlir.LLVMRelocStatic,
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
		MaxAlign: pointerAlignment,
	}

	// Collect dependency directories from the first packages.Load so that
	// scanCGoFiles can process import "C" in dependency packages too.
	depDirs := make(map[string]string, len(allPkgs))
	for _, pkg := range allPkgs {
		if pkg.PkgPath != "" && pkg.Dir != "" {
			depDirs[pkg.PkgPath] = pkg.Dir
		}
	}

	// Compute sysroot include path for CGo preprocessing.
	sigoRoot := options.Environment.Value("SIGOROOT")
	abi := strings.Split(triplet, "-")[0]
	if fpuEnabled {
		abi += "-fp"
	} else {
		abi += "-nofp"
	}
	var cIncludePaths []string
	sysrootInclude := filepath.Join(sigoRoot, "sysroots", abi, "include")
	if _, err := os.Stat(sysrootInclude); err == nil {
		cIncludePaths = append(cIncludePaths, sysrootInclude)
	}

	// Ask the C compiler for its resource directory so we can find builtin
	// headers like stddef.h and stdarg.h.
	if cc, err := findToolchain(options.Environment); err == nil {
		if out, err := exec.Command(cc.CC, "-print-resource-dir").Output(); err == nil {
			builtinInclude := filepath.Join(strings.TrimSpace(string(out)), "include")
			if _, err := os.Stat(builtinInclude); err == nil {
				cIncludePaths = append(cIncludePaths, builtinInclude)
			}
		}
	}

	// Create a new program.
	program := ssa.NewProgram(&ssa.ProgramConfig{
		Tags: tags,
		//AdditionalPackages: additionalPackages,
		Environment:    options.Environment.List(),
		PackagePath:    packageDir,
		ModuleRoot:     moduleDir,
		GoRoot:         options.Environment.Value("GOROOT"),
		Sizes:          &sizes,
		DependencyDirs: depDirs,
		TargetTriplet:  triplet,
		CpuName:        options.Cpu,
		IncludePaths:   cIncludePaths,
	})

	// Parse the package.
	fmt.Print("Parsing packages...")
	phaseStart := time.Now()
	if err := program.Parse(ctx); err != nil {
		// TODO: Replace paths from the virtual GOROOT with the real paths in error strings.
		return err
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

	// Initialize MLIR.
	mlirCtx := mlir.NewContext()
	mlirCtx.RegisterAllLLVMTranslations()
	goir.DialectHandle().RegisterDialect(mlirCtx)
	mlirCtx.LoadAllAvailableDialects()

	mlir.RegisterAllPasses()

	// Create the MLIR module.
	mlirModule := mlir.NewModule(mlir.NewUnknownLoc(mlirCtx))

	// Set module attributes before creating the SSA builder.
	goir.SetTargetDataLayout(mlirModule, targetLayout)
	goir.SetTargetTriple(mlirModule, triplet)

	// Compile C preamble to an object file using clang -c.
	// Preambles were extracted during Parse via the CGo overlay pre-scan.
	var preambleObjFile string
	var preambleBuf strings.Builder
	// Prepend #define directives from #cgo CFLAGS: -D... pragmas.
	for _, def := range program.CGODefines {
		eqIdx := strings.IndexByte(def, '=')
		if eqIdx >= 0 {
			fmt.Fprintf(&preambleBuf, "#define %s %s\n", def[:eqIdx], def[eqIdx+1:])
		} else {
			fmt.Fprintf(&preambleBuf, "#define %s\n", def)
		}
	}

	for _, p := range program.CGoPreambles {
		if p.GoFile != "" {
			fmt.Fprintf(&preambleBuf, "#line %d \"%s\"\n", p.GoLine, p.GoFile)
		}
		preambleBuf.WriteString(p.Text)
		preambleBuf.WriteByte('\n')
	}

	if preamble := preambleBuf.String(); strings.TrimSpace(preamble) != "" {
		fmt.Print("Compiling C preamble...")

		// Append non-static wrappers for any static inline functions
		// referenced from Go code.
		if len(program.CGOStaticFuncs) > 0 {
			preamble += ssa.GenerateStaticWrappers(program.CGOStaticFuncs, program.CGOFuncNames)
		}

		// Write preamble to a temp .c file.
		preambleC := filepath.Join(options.BuildDir, "preamble.c")
		if err := os.WriteFile(preambleC, []byte(preamble), 0644); err != nil {
			return errors.Join(ErrCodeGeneratorError, fmt.Errorf("writing preamble: %w", err))
		}

		preambleObjFile = filepath.Join(options.BuildDir, "preamble.o")
		preambleIncludePaths := append(cIncludePaths, program.CGOIncludePaths...)

		clangArgs := clangTargetFlags(
			triplet,
			cpuType,
			fpuType,
			options.Float,
		)

		clangArgs = append(
			clangArgs,
			"-c", preambleC,
			"-o", preambleObjFile,
			"-fno-builtin",
		)

		for _, inc := range preambleIncludePaths {
			clangArgs = append(clangArgs, "-I"+inc)
		}
		if options.GenerateDebugInfo {
			clangArgs = append(clangArgs, "-g")
		}

		// Find the C compiler.
		tc, err := findToolchain(options.Environment)
		if err != nil {
			return errors.Join(ErrCodeGeneratorError, fmt.Errorf("finding toolchain for preamble: %w", err))
		}
		clangCmd := exec.Command(tc.CC, clangArgs...)
		clangCmd.Stderr = os.Stderr
		if err := clangCmd.Run(); err != nil {
			fmt.Println()
			fmt.Println("Command failed:", clangCmd.String())
			return errors.Join(ErrCodeGeneratorError, fmt.Errorf("C preamble compilation failed: %w", err))
		}
		fmt.Println("done")
	}

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
	phaseStart = time.Now()
	if err := builder.GeneratePackages(ctx, program.OrderedPackages); err != nil {
		fmt.Printf("fail (%.2fs)\n", time.Since(phaseStart).Seconds())
		return errors.Join(ErrCodeGeneratorError, fmt.Errorf("SSA generation failed: %w", err))
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

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
	phaseStart = time.Now()
	if runOptimizerPass(mlirModule, options.DebugLowering).IsFailure() {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err, errors.New("optimization passes failed"))
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

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
	phaseStart = time.Now()
	llvmModule := mlir.TranslateModuleToLLVMIR(mlirModule.Operation(), llvmContext)

	if !options.GenerateDebugInfo {
		// Strip debug info
		llvmModule.StripModuleDebugInfo()
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

	// Add required constant globals to the LLVM module directly
	addConstantGlobals(llvmModule, options, fpuEnabled, uint(stackAlignment), targetLayout)

	if options.DumpIR {
		err := dumpModule(llvmModule, options.Output+".dump.ll")
		if err != nil {
			return err
		}
	}

	// Optimize modules
	fmt.Print("Optimizing LLVM IR...")
	phaseStart = time.Now()
	if err = optimize(llvmModule, options.Optimization, machine); err != nil {
		fmt.Println()
		return errors.Join(ErrCodeGeneratorError, err)
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

	if options.DumpIR {
		err := dumpModule(llvmModule, options.Output+".dump.opt.ll")
		if err != nil {
			return err
		}
	}

	fmt.Print("Linking firmware image...")
	phaseStart = time.Now()
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
		preambleObj:   preambleObjFile,
	}, options,
	); err != nil {
		fmt.Println()
		return err
	}
	fmt.Printf("done (%.2fs)\n", time.Since(phaseStart).Seconds())

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
	preambleObj   string // path to compiled C preamble object file (empty if none)
	ramRegion     string
	heapRegion    string
	stackRegion   string
	linkerDefines map[string]string
}

func link(options linkOptions, buildOptions BuildOptions) error {
	sigoRoot := buildOptions.Environment.Value("SIGOROOT")

	// Determine sysroot.
	abi := strings.Split(options.triplet, "-")[0]
	if options.floatEnabled {
		abi += "-fp"
	} else {
		abi += "-nofp"
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

	if len(options.prog.LinkerScript) == 0 {
		return errors.New("no linker script found")
	}

	// Preprocess the linker script.
	linkerFilePath := filepath.Join(buildOptions.BuildDir, "linker.ld")
	if err := preprocess(preprocessParameters{
		toolchain:    toolchain,
		triplet:      options.triplet,
		includePaths: collectLinkerIncludePaths(options),
		defines:      collectLinkerDefines(options),
		input:        options.prog.LinkerScript,
		output:       linkerFilePath,
	}); err != nil {
		return err
	}

	// Other arguments
	elfOut := filepath.Join(buildOptions.BuildDir, "package.elf")
	args := []string{
		"--sysroot=" + sysroot,
		"--defsym=sigo_headGoroutine=runtime.headGoroutine",
		"--defsym=sigo_currentGoroutine=runtime.currentGoroutine",
		"--defsym=sigo_goroutineStackSize=runtime._goroutineStackSize",
		"--defsym=sigo_coroStackSize=runtime._coroStackSize",
		"-v",
		"--gc-sections",
		"-o", elfOut,
		"-nostdlib",
		"-L" + filepath.Join(sysroot, "lib"),
		"-L" + filepath.Join(sigoRoot, "runtime"),
		"-L" + filepath.Dir(options.prog.LinkerScript),
		"-T" + filepath.Join(buildOptions.BuildDir, "linker.ld"),
	}

	if buildOptions.GenerateDebugInfo {
		args = append(args, "-g")
	}

	// Add all linker files
	for _, ld := range append(options.prog.Files[".ld"], options.prog.Files[".linker"]...) {
		args = append(args, "-L"+filepath.Dir(ld))
	}

	// Add the main firmware compiled object.
	args = append(args, objectOut)

	if options.preambleObj != "" {
		args = append(args, options.preambleObj)
	}

	// Compile assembly and append every resulting object before libraries.
	for _, asm := range append(options.prog.Files[".s"], options.prog.Files[".asm"]...) {
		fname, err := filepath.EvalSymlinks(asm)
		if err != nil {
			return errors.Join(ErrCompilerFailed, err)
		}

		objFile := filepath.Join(
			buildOptions.BuildDir,
			fmt.Sprintf("%s-%d.o", filepath.Base(asm), rand.Int()),
		)

		assemblerArgs := clangTargetFlags(
			options.triplet,
			options.cpu,
			options.fpu,
			buildOptions.Float,
		)

		assemblerArgs = append(
			assemblerArgs,
			"-c", fname,
			"-o", objFile,
		)

		if buildOptions.GenerateDebugInfo {
			assemblerArgs = append(assemblerArgs, "-g")
		}

		for def, val := range options.prog.Defines {
			if val == "" {
				assemblerArgs = append(assemblerArgs, "-D"+def)
			} else {
				assemblerArgs = append(
					assemblerArgs,
					fmt.Sprintf("-D%s=%s", def, val),
				)
			}
		}

		clangCmd := exec.Command(toolchain.CC, assemblerArgs...)
		clangCmd.Stderr = os.Stderr

		if err := clangCmd.Run(); err != nil {
			fmt.Println()
			fmt.Println("Command failed:", clangCmd.String())
			return errors.Join(ErrCompilerFailed, err)
		}

		args = append(args, objFile)
	}

	// Libraries come after every object.
	args = append(args, "-lc", "-lclang_rt.builtins")

	// Add CGo LD flags.
	args = append(args, options.prog.CGOLDFlags...)

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
		passes = "default<Os>"
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

func addConstantGlobals(module mlir.LLVMModuleRef, options BuildOptions, floatEnabled bool, stackAlignment uint, dataLayout mlir.LLVMTargetDataRef) {
	ctx := module.Context()
	intPtrType := ctx.IntPtrType(dataLayout)
	boolType := ctx.Int1Type()

	// Stack size for goroutines
	globalGoroutineStackSize := findOrCreateGlobal(module, intPtrType, "runtime._goroutineStackSize")
	constGoroutineStackSize := mlir.NewConstInt(intPtrType, uint64(alignUp(uint(options.StackSize), stackAlignment)), false)
	globalGoroutineStackSize.SetAlignment(stackAlignment)
	globalGoroutineStackSize.SetInitializer(constGoroutineStackSize)
	globalGoroutineStackSize.SetLinkage(mlir.LLVMLinkageExternal)
	globalGoroutineStackSize.SetGlobalConstant(true)

	// Default stack size for coroutines (used by iter.Pull / runtime.newcoro).
	coroStackSize := options.CoroStackSize
	if coroStackSize <= 0 {
		coroStackSize = options.StackSize
	}
	globalCoroStackSize := findOrCreateGlobal(module, intPtrType, "runtime._coroStackSize")
	constCoroStackSize := mlir.NewConstInt(intPtrType, uint64(alignUp(uint(coroStackSize), stackAlignment)), false)
	globalCoroStackSize.SetAlignment(stackAlignment)
	globalCoroStackSize.SetInitializer(constCoroStackSize)
	globalCoroStackSize.SetLinkage(mlir.LLVMLinkageExternal)
	globalCoroStackSize.SetGlobalConstant(true)

	// FPU enable flag.
	globalFpuEnableFlag := findOrCreateGlobal(module, boolType, "runtime._fpuEnabled")
	alignment := dataLayout.PreferredAlignmentOfGlobal(globalFpuEnableFlag)

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

func alignUp(n, alignment uint) uint {
	if alignment == 0 {
		panic("alignment must be nonzero")
	}

	remainder := n % alignment
	if remainder == 0 {
		return n
	}

	return n + alignment - remainder
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

// goarch maps a .td arch value to a valid Go GOARCH value.
// The .td files use LLVM-oriented names (e.g. "thumb2") that don't
// correspond to Go's architecture identifiers.
func goarch(arch string) string {
	switch strings.ToLower(arch) {
	case "arm", "thumb2":
		return "arm"
	case "riscv64":
		return "riscv64"
	default:
		return strings.ToLower(arch)
	}
}

// extractCGoPreambles collects the C preamble text from all import "C"
// declarations across the ordered package list. Multiple preambles are
// concatenated so they can be compiled together in a single CIR pass.
