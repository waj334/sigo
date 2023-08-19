package ssa

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/packages"

	"omibyte.io/sigo/mlir"
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
			mlirCtx := mlir.ContextCreate()
			mlir.DialectHandleRegisterDialect(mlir.GetDialectHandle__go__(), mlirCtx)
			mlir.ContextLoadAllAvailableDialects(mlirCtx)

			// Destroy the context at the end of the test.
			defer mlir.ContextDestroy(mlirCtx)

			// Set up type checker.
			testImporter := &TestImporter{
				pkgs:            map[string]*types.Package{},
				defaultImporter: importer.Default(),
			}
			config := types.Config{Importer: testImporter}
			info := &types.Info{
				Types:      map[ast.Expr]types.TypeAndValue{},
				Instances:  map[*ast.Ident]types.Instance{},
				Defs:       map[*ast.Ident]types.Object{},
				Uses:       map[*ast.Ident]types.Object{},
				Implicits:  map[ast.Node]types.Object{},
				Selections: map[*ast.SelectorExpr]*types.Selection{},
				Scopes:     map[ast.Node]*types.Scope{},
				InitOrder:  []*types.Initializer{},
			}

			fset := token.NewFileSet()

			// Parse the runtime package.
			runtimeFile, err := parser.ParseFile(fset, "./_testdata/src/runtime/runtime.go", nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("Failed to parse the test runtime: %v", err)
			}

			// Parse the "somepkg" package which is used to test interaction with external packages.
			somePkgFile, err := parser.ParseFile(fset, "./_testdata/src/somepkg/decl.go", nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("Failed to parse the test runtime: %v", err)
			}

			volatileFile, err := parser.ParseFile(fset, "./_testdata/src/volatile/doc.go", nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("Failed to parse the test runtime: %v", err)
			}

			// Parse the source file.
			parsedFile, err := parser.ParseFile(fset, sourceFile, nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("Failed to parse the source file: %v", err)
			}

			// Type check the parsed packages.
			runtimePkg, err := config.Check("runtime", fset, []*ast.File{runtimeFile}, info)
			if err != nil {
				t.Fatal(err)
			}
			testImporter.pkgs[runtimePkg.Path()] = runtimePkg

			somePkg, err := config.Check("somepkg", fset, []*ast.File{somePkgFile}, info)
			if err != nil {
				t.Fatal(err)
			}
			testImporter.pkgs[somePkg.Path()] = somePkg

			volatilePkg, err := config.Check("volatile", fset, []*ast.File{volatileFile}, info)
			if err != nil {
				t.Fatal(err)
			}
			testImporter.pkgs[volatilePkg.Path()] = volatilePkg

			srcPkg, err := config.Check("main", fset, []*ast.File{parsedFile}, info)
			if err != nil {
				t.Fatal(err)
			}

			pkg := &packages.Package{
				Name:    srcPkg.Name(),
				PkgPath: srcPkg.Path(),
				Imports: map[string]*packages.Package{
					"runtime":  createPackage(runtimePkg, fset, info, []*ast.File{runtimeFile}),
					"somepkg":  createPackage(somePkg, fset, info, []*ast.File{somePkgFile}),
					"volatile": createPackage(volatilePkg, fset, info, []*ast.File{volatileFile}),
				},
				Types:     srcPkg,
				Fset:      fset,
				Syntax:    []*ast.File{parsedFile},
				TypesInfo: info,
				TypesSizes: &types.StdSizes{
					WordSize: 4,
					MaxAlign: 4,
				},
			}

			// Create the SSA Program.
			program := NewProgram(&ProgramConfig{
				Tags:               nil,
				AdditionalPackages: nil,
				Environment:        nil,
				PackagePath:        "",
			})

			// Add the packages to the program.
			program.AddPackage(pkg)

			builder := NewBuilder(Config{
				NumWorkers: 1,
				Fset:       fset,
				Ctx:        mlirCtx,
				Info:       info,
				Sizes: &types.StdSizes{
					WordSize: 4,
					MaxAlign: 4,
				},
				Module:  mlir.ModuleCreateEmpty(mlir.LocationUnknownGet(mlirCtx)),
				Program: program,
			})

			// Build SSA for parsed file.
			ctx := context.Background()
			builder.GeneratePackages(ctx, []*packages.Package{pkg.Imports["somepkg"], pkg})

			// Dump the module to a string.
			inputText := mlir.ModuleDump(builder.config.Module)

			//Verify the module.
			if mlir.LogicalResultIsFailure(mlir.VerifyModule(builder.config.Module)) {
				t.Error("Module verification failed")
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

// Function to run single command with input
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
