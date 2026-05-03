package ssa

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"pkg.si-go.dev/go-mlir/mlir"
	"pkg.si-go.dev/sigo/goir/binding/goir"
)

var enabledTests string

type TestImporter struct {
	pkgs            map[string]*types.Package
	defaultImporter types.Importer
}

func (t *TestImporter) Import(path string) (*types.Package, error) {
	if pkg, ok := t.pkgs[path]; ok {
		return pkg, nil
	}
	return t.defaultImporter.Import(path)
}

func TestSSA(t *testing.T) {
	var tests []string
	if len(enabledTests) > 0 {
		tests = strings.Split(enabledTests, ",")
		fmt.Printf("Enabled tests: [%+v]\n", enabledTests)
		println("Number of tests: ", len(tests))
	}

	sourceFiles, _ := filepath.Glob("./_testdata/*.go")
	for _, sourceFile := range sourceFiles {
		disabled := false
		t.Run(sourceFile, func(t *testing.T) {
			if strings.Contains(sourceFile, "_disabled") {
				disabled = true
			}

			if len(tests) > 0 {
				filename := filepath.Base(sourceFile)
				disabled = !slices.Contains(tests, filename)
			}

			if disabled {
				t.Skip("disabled")
			}

			// Initialize MLIR.
			mlirCtx := mlir.NewContext()
			goir.DialectHandle().RegisterDialect(mlirCtx)
			mlirCtx.LoadAllAvailableDialects()

			// Destroy the context at the end of the test.
			defer mlirCtx.Destroy()

			// Set up type checker.
			testImporter := &TestImporter{
				pkgs:            map[string]*types.Package{},
				defaultImporter: importer.Default(),
			}
			config := types.Config{Importer: testImporter}

			fset := token.NewFileSet()

			// Parse the SiGo runtime package.
			runtimeGoFiles, err := filepath.Glob("../../src/runtime/*.go")
			if err != nil {
				t.Fatalf("Failed to glob runtime files: %v", err)
			}
			runtimeFiles := make([]*ast.File, 0, len(runtimeGoFiles))
			for _, goFile := range runtimeGoFiles {
				f, err := parser.ParseFile(fset, goFile, nil, parser.ParseComments)
				if err != nil {
					t.Fatalf("Failed to parse runtime file %s: %v", goFile, err)
				}
				runtimeFiles = append(runtimeFiles, f)
			}

			// Type check the runtime package.
			runtimeInfo := types.Info{
				Types:      map[ast.Expr]types.TypeAndValue{},
				Instances:  map[*ast.Ident]types.Instance{},
				Defs:       map[*ast.Ident]types.Object{},
				Uses:       map[*ast.Ident]types.Object{},
				Implicits:  map[ast.Node]types.Object{},
				Selections: map[*ast.SelectorExpr]*types.Selection{},
				Scopes:     map[ast.Node]*types.Scope{},
				InitOrder:  []*types.Initializer{},
			}

			runtimePkg, err := config.Check("runtime", fset, runtimeFiles, &runtimeInfo)
			if err != nil {
				t.Fatal(err)
			}
			testImporter.pkgs[runtimePkg.Path()] = runtimePkg

			// The SSA builder iterates every decl in the runtime package
			// (use-analysis is bypassed for runtime — see builder.go:458). To
			// keep the builder from following selectors into packages outside
			// the SSA program (os, sync, internal/chacha8rand, ...), drop
			// function bodies, top-level var decls (e.g. headGoroutine, whose
			// type transitively references chacha8rand.State), and consts.
			// Type definitions are kept — they're what the SSA builder
			// actually consumes via Program.LookupType.
			for _, f := range runtimeFiles {
				kept := f.Decls[:0]
				for _, decl := range f.Decls {
					switch d := decl.(type) {
					case *ast.FuncDecl:
						d.Body = nil
						kept = append(kept, d)
					case *ast.GenDecl:
						if d.Tok == token.VAR || d.Tok == token.CONST {
							continue
						}
						kept = append(kept, d)
					default:
						kept = append(kept, decl)
					}
				}
				f.Decls = kept
			}
			// InitOrder still references the dropped var initializers; clear
			// it so GeneratePackages doesn't try to emit globals for them.
			runtimeInfo.InitOrder = nil

			// Parse the source file.
			parsedFile, err := parser.ParseFile(fset, sourceFile, nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("Failed to parse the source file: %v", err)
			}

			// Parse the imported packages.
			imports := make(map[string]*packages.Package, len(parsedFile.Imports)+1)
			imports["runtime"] = createPackage(runtimePkg, fset, &runtimeInfo, runtimeFiles)
			for _, importSpec := range parsedFile.Imports {
				pkgPath := strings.Trim(importSpec.Path.Value, "\"")
				pkgName := pkgPath
				if last := strings.LastIndex(pkgPath, "/"); last >= 0 {
					pkgName = pkgPath[last+1:]
				}

				if pkgName == "unsafe" {
					// Skip the unsafe package since it is special.
					continue
				}

				pkgDir := filepath.Join("./_testdata/src", pkgName, "*.go")
				goFiles, err := filepath.Glob(pkgDir)
				if err != nil {
					t.Fatal(err)
				}

				// Parse the test package's sources.
				files := make([]*ast.File, len(goFiles))
				for i, goFile := range goFiles {
					f, err := parser.ParseFile(fset, goFile, nil, parser.ParseComments)
					if err != nil {
						t.Fatal(err)
					}
					files[i] = f
				}

				// Construct and type check the test package.
				info := types.Info{
					Types:      map[ast.Expr]types.TypeAndValue{},
					Instances:  map[*ast.Ident]types.Instance{},
					Defs:       map[*ast.Ident]types.Object{},
					Uses:       map[*ast.Ident]types.Object{},
					Implicits:  map[ast.Node]types.Object{},
					Selections: map[*ast.SelectorExpr]*types.Selection{},
					Scopes:     map[ast.Node]*types.Scope{},
					InitOrder:  []*types.Initializer{},
				}
				checked, err := config.Check(pkgPath, fset, files, &info)
				if err != nil {
					t.Fatal(err)
				}
				testImporter.pkgs[checked.Path()] = checked
				imports[checked.Path()] = createPackage(checked, fset, &info, files)
			}

			// Finally, type check the test main package.
			srcInfo := types.Info{
				Types:      map[ast.Expr]types.TypeAndValue{},
				Instances:  map[*ast.Ident]types.Instance{},
				Defs:       map[*ast.Ident]types.Object{},
				Uses:       map[*ast.Ident]types.Object{},
				Implicits:  map[ast.Node]types.Object{},
				Selections: map[*ast.SelectorExpr]*types.Selection{},
				Scopes:     map[ast.Node]*types.Scope{},
				InitOrder:  []*types.Initializer{},
			}
			srcPkg, err := config.Check("main", fset, []*ast.File{parsedFile}, &srcInfo)
			if err != nil {
				t.Fatal(err)
			}

			sizes := types.StdSizes{
				WordSize: 4,
				MaxAlign: 4,
			}

			pkg := &packages.Package{
				Name:       srcPkg.Name(),
				PkgPath:    srcPkg.Path(),
				Imports:    imports,
				Types:      srcPkg,
				Fset:       fset,
				Syntax:     []*ast.File{parsedFile},
				TypesInfo:  &srcInfo,
				TypesSizes: &sizes,
			}

			// Create the SSA Program.
			program := NewProgram(&ProgramConfig{
				Tags:               nil,
				AdditionalPackages: nil,
				Environment:        nil,
				PackagePath:        "",
				Sizes:              &sizes,
			})

			// Add the packages to the program.
			program.AddPackage(pkg)

			// Compute dependency graph.
			err = program.computePackageOrder()
			if err != nil {
				t.Fatal(err)
			}

			builder := NewBuilder(Config{
				NumWorkers: 1,
				Fset:       fset,
				Ctx:        mlirCtx,
				Sizes: &types.StdSizes{
					WordSize: 4,
					MaxAlign: 4,
				},
				Module:             mlir.NewModule(mlir.NewUnknownLoc(mlirCtx)),
				Program:            program,
				DisableUseAnalysis: true,
			})

			// Build SSA for parsed file.
			ctx := context.Background()
			builder.GeneratePackages(ctx, program.OrderedPackages)

			// Dump the module to a string.
			inputText := builder.config.Module.Operation().String()

			//Verify the module.
			if !builder.config.Module.Operation().Verify() {
				t.Errorf("Module verification failed: %v\n\n%s\n\n----------------------\n\n", err, inputText)
			}

			// Get the command from the comment on the first line.
			if len(parsedFile.Comments) == 0 {
				t.Fatal("Missing command")
			}
			command := parsedFile.Comments[0].Text()

			if !strings.Contains(command, "RUN:") {
				t.Fatal("Missing command")
			}

			// Trim the command.
			command = strings.TrimSpace(strings.ReplaceAll(command, "RUN:", ""))

			// Substitute in the source file for %s.
			command = strings.ReplaceAll(command, "%s", sourceFile)

			// Run the command with the IR.
			err = runCommand(command, inputText)
			if err != nil {
				t.Errorf("Failed to run command from source file: %v\n\n%s\n\n----------------------\n\n", err, inputText)
			}
		})
	}
}

// Function to run a single command with input
func runCommand(command string, input string) error {
	var cmd *exec.Cmd

	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", command)
	} else {
		cmd = exec.Command("/bin/sh", "-c", command)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Create a buffer and assign it to cmd.Stdin
	var buf bytes.Buffer
	buf.WriteString(input)
	cmd.Stdin = &buf

	err := cmd.Run()
	if err != nil {
		return err
	}

	return nil
}

func createPackage(pkg *types.Package, fset *token.FileSet, info *types.Info, files []*ast.File) *packages.Package {
	return &packages.Package{
		Name:      pkg.Name(),
		PkgPath:   pkg.Path(),
		Types:     pkg,
		Fset:      fset,
		Syntax:    files,
		TypesInfo: info,
		TypesSizes: &types.StdSizes{
			WordSize: 4,
			MaxAlign: 4,
		},
	}
}

func TestMain(m *testing.M) {
	flag.StringVar(&enabledTests, "tests", "", "Run only the specified tests.")
	flag.Parse()
	os.Exit(m.Run())
}
