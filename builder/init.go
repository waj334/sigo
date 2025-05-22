package builder

import "pkg.si-go.dev/sigo/llvm"

func init() {
	// Set up compile target
	llvm.InitializeAllTargets()
	llvm.InitializeAllTargetInfos()
	llvm.InitializeAllTargetMCs()
	llvm.InitializeAllAsmParsers()
	llvm.InitializeAllAsmPrinters()
}
