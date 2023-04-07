package compiler

import (
	"context"
	"omibyte.io/sigo/llvm"
	"strings"
	"testing"

	"omibyte.io/sigo/compiler/testutil"
)

func init() {
	// Set up compile target
	llvm.InitializeAllTargets()
	llvm.InitializeAllTargetInfos()
	llvm.InitializeAllTargetMCs()
	llvm.InitializeAllAsmParsers()
	llvm.InitializeAllAsmPrinters()
}

func TestNoReturn(t *testing.T) {
	prog, err := testutil.CompileTestProgram(`
	package main
	func testFunc() {
		i := 0
		println(i)
	}
	`)
	if err != nil {
		t.Fatal(err)
	}

	target, err := NewTargetFromMap(map[string]string{
		"architecture": "arm",
		"cpu":          "cortex-m4",
		"triple":       "thumb-none-eabi",
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := target.Initialize(); err != nil {
		t.Fatal(err)
	}

	cc := NewCompiler(target)

	// Don't generate debug info for the test
	cc.GenerateDebugInfo = true

	// Compile the test input
	for _, pkg := range prog.AllPackages() {
		if err = cc.CompilePackage(context.Background(), pkg); err != nil {
			t.Fatal(err)
		}
	}

	result := llvm.PrintModuleToString(cc.Module())

	expected := `define void @init() !dbg !0 {
init.entry:
}

define void @testFunc() !dbg !5 {
test.entry:
}
`

	if !strings.Contains(result, strings.TrimSpace(expected)) {
		t.Errorf("expected result to contain:\n%+v\n\nGot:\n%v", expected, result)
	}

	println(result)
}
