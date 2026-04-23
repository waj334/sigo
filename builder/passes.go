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
		flags := mlir.NewOpPrintingFlags().
			WithEnableDebugInfo(true, true).
			WithPrintNameLocAsPrefix()
		defer flags.Destroy()

		pm.EnableIRPrinting(mlir.IRPrinterConfig{
			PrintBeforeAll: true,
			PrintAfterAll:  true,
			Flags:          flags,
		}).
			EnableStatistics(mlir.PassDisplayModePipeline).
			EnableTiming()
	}

	pm.NestedUnder("go.func").
		AddOwnedPass(goir.NewValueNormalizationFuncPass()).
		AddOwnedPass(goir.NewEliminateRedundantNilChecksPass())

	pm.AddOwnedPass(goir.NewPreprocessingPass()).
		AddOwnedPass(goir.NewCallPass()).
		AddOwnedPass(goir.NewGlobalConstantsPass())

	// Canonicalize following NewGlobalConstantsPass so that constant expressions are folded.
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer())

	pm.NestedUnder("go.global").
		AddOwnedPass(goir.NewValueNormalizationGlobalPass())

	pm.AddOwnedPass(goir.NewGlobalInitializerPass()).
		AddOwnedPass(goir.NewImmutableGlobalsPass())

	pm.NestedUnder("go.global").
		AddOwnedPass(goir.NewAttachDebugInfoToGlobalPass())

	// Handle heap and stack allocations after lowering globals.
	pm.NestedUnder("go.func").
		AddOwnedPass(goir.NewHeapEscapePass()).
		AddOwnedPass(goir.NewFuncPass()).
		AddOwnedPass(goir.NewInsertGCWriteBarrierPass()).
		AddOwnedPass(goir.NewAttachDebugInfoToFuncPass()).
		NestedUnder("go.alloca").
		AddOwnedPass(goir.NewAttachDebugInfoToAllocaPass())

	// Lower GoIR dialect to builtin dialects.
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer()).
		AddOwnedPass(goir.NewLowerToCorePass())

	// Lower to LLVMIR.
	pm.AddOwnedPass(mlir.NewTransformsCanonicalizer()).
		AddOwnedPass(goir.NewLowerToLLVMPass())

	// Attach debug info to CIR-derived llvm.func ops that lack it.
	// GoIR-derived functions already have DISubprogramAttr and are skipped.
	pm.NestedUnder("llvm.func").
		AddOwnedPass(goir.NewAttachDebugInfoToLLVMFuncPass())

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
