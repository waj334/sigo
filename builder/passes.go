package builder

import (
	"fmt"

	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

func runOptimizerPass(module mlir.Module, debug bool) mlir.LogicalResult {
	//moduleOpName := "builtin.module"
	moduleOpName := module.Operation().Name().String()
	pm := mlir.NewPassManagerOnOperation(module.Context(), moduleOpName).
		EnableVerifier(true)

	if debug {
		module.Context().EnableMultithreading(false)
		pm.EnableIRPrinting(mlir.IRPrinterConfig{}).
			EnableStatistics(mlir.PassDisplayModePipeline).
			EnableTiming()
	}

	pm.NestedUnder("go.func").
		AddOwnedPass(goir.NewValueNormalizationFuncPass()).
		AddOwnedPass(goir.NewFuncPass()).
		AddOwnedPass(goir.NewAttachDebugInfoToFuncPass()).
		NestedUnder("go.alloca").
		AddOwnedPass(goir.NewAttachDebugInfoToAllocaPass())

	pm.NestedUnder("go.global").
		AddOwnedPass(goir.NewValueNormalizationGlobalPass()).
		AddOwnedPass(goir.NewAttachDebugInfoToGlobalPass())

	pm.AddOwnedPass(goir.NewPreprocessingPass()).
		AddOwnedPass(goir.NewCallPass()).
		AddOwnedPass(goir.NewGlobalConstantsPass()).
		AddOwnedPass(goir.NewGlobalInitializerPass())

	// Handle heap and stack allocations after lowering globals.
	pm.NestedUnder("go.func").
		AddOwnedPass(goir.NewHeapEscapePass())

	// Lower GoIR dialect to builtin dialects.
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer()).
		AddOwnedPass(goir.NewLowerToCorePass())

	// Lower to LLVMIR.
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer()).
		AddOwnedPass(goir.NewLowerToLLVMPass())

	// LLVM export and cleanup.
	pm.NestedUnder("llvm.func").
		AddOwnedPass(mlir.NewLLVMLegalizeForExportPass())
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer()).
		AddOwnedPass(mlir.NewTransformsSymbolDCE()).
		AddOwnedPass(mlir.NewTransformsCanonicalizer())

	if debug {
		// Print the pipeline.
		fmt.Printf("Pipeline:\n%s\n", pm.AsOpPassManager().String())
	}

	return pm.Run(module.Operation())
}
