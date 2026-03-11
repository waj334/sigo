package builder

import "pkg.si-go.dev/go-mlir/mlir"

func init() {
	// Set up compile target
	mlir.LLVMInitializeAllTargets()
	mlir.LLVMInitializeAllTargetInfos()
	mlir.LLVMInitializeAllTargetMCs()
	mlir.LLVMInitializeAllAsmParsers()
	mlir.LLVMInitializeAllAsmPrinters()
}
