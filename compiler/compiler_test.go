package compiler

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"omibyte.io/sigo/llvm"
	"testing"
)

func init() {
	// Set up compile target
	llvm.InitializeAllTargets()
	llvm.InitializeAllTargetInfos()
	llvm.InitializeAllTargetMCs()
	llvm.InitializeAllAsmParsers()
	llvm.InitializeAllAsmPrinters()
}

func TestExpressions(t *testing.T) {
	tests := []struct {
		name string
		src  string
		ir   string
	}{
		{
			"alloc", `
			package p
			
			func use(int){}
			func f00() {
				var x int
				use(x)
			}
			`,
			"",
		},
	}

	// Run each test case
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var ssaPkg *ssa.Package
			t.Parallel()
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "p.go", tc.src, 0)
			if err != nil {
				t.Error(err)
			}
			files := []*ast.File{f}

			pkg := types.NewPackage("p", "")
			conf := &types.Config{Importer: nil}
			ssaPkg, _, err = ssautil.BuildPackage(conf, fset, pkg, files, ssa.SanityCheckFunctions|ssa.BareInits|ssa.GlobalDebug|ssa.InstantiateGenerics|ssa.NaiveForm)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			t.Log("Package parsed")

			// Get the target
			target, err := NewTargetFromMap(map[string]string{
				"architecture": "arm",
				"cpu":          "cortex-m4",
				"triple":       "armv7m-none-eabi",
			})

			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// Initialize the target
			if err = target.Initialize(); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			t.Log("Target initialized")

			// Create a compiler for this test
			cc, ctx := NewCompiler(tc.name, NewOptions().WithTarget(target))

			// Compile the package
			err = cc.CompilePackage(context.Background(), ctx, ssaPkg)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			cc.Finalize()

			t.Log("Package compiled")

			// Strip the debug info
			llvm.StripModuleDebugInfo(cc.module)

			// Get the IR
			ir := llvm.PrintModuleToString(cc.module)

			cc.Dispose()
			target.Dispose()

			if ir != tc.ir {
				t.Errorf("expected:\n%s\n\ngot:\n%s\n\n", tc.ir, ir)
			}
		})
	}
}
