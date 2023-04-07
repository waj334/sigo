package testutil

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
)

type panicImporter struct{}

func (p panicImporter) Import(path string) (*types.Package, error) {
	panic(fmt.Errorf("import not allowed: %s", path))
}

func CompileTestProgram(programStr string) (*ssa.Program, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", programStr, 0)
	if err != nil {
		return nil, fmt.Errorf("error parsing source: %v\n", err)
	}

	mainPkg := types.NewPackage("main", "main")

	conf := loader.Config{
		Fset: fset,
		TypeChecker: types.Config{
			Importer: &panicImporter{},
		},
	}

	conf.CreateFromFiles("main", file)
	//prog, _ := conf.Load()
	ssaProg := ssa.NewProgram(fset, ssa.BuildSerially|ssa.BareInits|ssa.GlobalDebug|ssa.InstantiateGenerics)

	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Instances:  map[*ast.Ident]types.Instance{},
	}

	if err = types.NewChecker(&conf.TypeChecker, fset, mainPkg, info).Files([]*ast.File{file}); err != nil {
		return nil, err
	}

	ssaProg.CreatePackage(mainPkg, []*ast.File{file}, info, false)
	ssaProg.Build()

	return ssaProg, nil
}

func PrintSSABlocks(prog *ssa.Program, fn string) {
	for _, pkg := range prog.AllPackages() {
		ssaFn := pkg.Func(fn)
		if ssaFn != nil {
			// Traverse the SSA representation and process the desired instructions
			for _, b := range ssaFn.Blocks {
				for _, instr := range b.Instrs {
					fmt.Println(instr)
				}
			}
		}
	}
}

func Filter[T ssa.Instruction](prog *ssa.Program, fn string) []T {
	var out []T
	for _, pkg := range prog.AllPackages() {
		ssaFn := pkg.Func(fn)
		if ssaFn != nil {
			// Traverse the SSA representation and process the desired instructions
			for _, b := range ssaFn.Blocks {
				for _, instr := range b.Instrs {
					if actual, ok := instr.(T); ok {
						out = append(out, actual)
					}
				}
			}
		}
	}

	return out
}
