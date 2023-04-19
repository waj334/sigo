package builder

import (
	"context"
	"errors"
	"go/ast"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"io/fs"
	"omibyte.io/sigo/compiler"
	"path/filepath"
	"strings"
)

type Program struct {
	config types.Config
	files  []*ast.File
	fset   *token.FileSet
	path   string
	env    Env

	pkgs []*packages.Package

	ssaProg    *ssa.Program
	targetInfo map[string]string

	options *compiler.Options

	targetLinkerFile      string
	targetLinkerSetByMain bool
	assemblyFiles         []string
}

func (p *Program) parse() (err error) {
	// Parse the package at the directory
	config := packages.Config{
		Mode:       packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedTypesSizes | packages.NeedModule | packages.NeedEmbedFiles | packages.NeedEmbedPatterns,
		Context:    context.Background(),
		Logf:       nil,
		Dir:        "",
		Env:        p.env.List(),
		BuildFlags: nil,
		Fset:       p.fset,
		ParseFile:  nil,
		Tests:      false,
	}

	pkgs, err := packages.Load(&config, "runtime", p.path)
	if err != nil {
		return err
	}

	allPackages := map[string]*packages.Package{}

	// Collect up all the packages
	for _, pkg := range pkgs {
		allPackages = gatherPackages(pkg, allPackages)
	}

	for _, pkg := range allPackages {
		// Check the syntax
		for _, syntaxError := range pkg.TypeErrors {
			err = errors.Join(err, syntaxError)
		}

		for _, syntaxError := range pkg.Errors {
			err = errors.Join(err, syntaxError)
		}

		if err != nil {
			continue
		}

		p.pkgs = append(p.pkgs, pkg)

		// Look in the packages for "target.ld". Prefer target.ld in the main
		// package. Otherwise, throw an error if more than one "target.ld" is
		// found. Also collect additional sources like assembly files TODO: C files too?
		if len(pkg.GoFiles) > 0 {
			pkgDir := filepath.Dir(pkg.GoFiles[0])
			filepath.Walk(pkgDir, func(path string, info fs.FileInfo, err error) error {
				if info.IsDir() && path != pkgDir {
					// Walk any subdirectories
					return filepath.SkipDir
				}

				ext := strings.ToLower(filepath.Ext(info.Name()))

				if info.Name() == "target.ld" && !p.targetLinkerSetByMain {
					if len(p.targetLinkerFile) == 0 || pkg.Name == "main" {
						p.targetLinkerFile = path
						if pkg.Name == "main" {
							// Main takes precedence. Stop considering other linker files
							p.targetLinkerSetByMain = true
						}
					} else {
						panic("multiple target linker files found")
					}
				} else if ext == ".s" || ext == ".asm" {
					p.assemblyFiles = append(p.assemblyFiles, path)
				}

				return nil
			})
		}

		// Gather files from packages and information from any pragma comments
		// found in each source.
		for _, file := range pkg.Syntax {
			p.files = append(p.files, file)
			// Process exported functions and globals
			for _, decl := range file.Decls {
				if fn, ok := decl.(*ast.FuncDecl); ok {
					symbolName := types.Id(pkg.Types, fn.Name.Name)
					info := p.options.GetSymbolInfo(symbolName)
					info.Exported = fn.Name.IsExported()

					// The main function should be exported as "main.main"
					if fn.Name.Name == "main" {
						info.LinkName = "main.main"
					}
				} else if global, ok := decl.(*ast.GenDecl); ok {
					for _, spec := range global.Specs {
						switch spec := spec.(type) {
						case *ast.ValueSpec:
							for _, name := range spec.Names {
								symbolName := types.Id(pkg.Types, name.Name)
								info := p.options.GetSymbolInfo(symbolName)
								info.Exported = name.IsExported()
							}
						case *ast.TypeSpec:
							symbolName := types.Id(pkg.Types, spec.Name.Name)
							info := p.options.GetSymbolInfo(symbolName)
							info.Exported = spec.Name.IsExported()
						}
					}
				}
			}

			// Process pragma comments
			for _, commentGroup := range file.Comments {
				for _, comment := range commentGroup.List {
					// Split the comment on the space character
					parts := strings.Split(comment.Text, " ")
					if count := len(parts); count > 1 {
						// Process the comment based of the first part
						switch parts[0] {
						//TODO: Only allow these in the main package
						case "//sigo:architecture":
							// value must follow
							if count == 2 {
								p.targetInfo["architecture"] = parts[1]
							} else {
								// TODO: Return syntax error
							}
						case "//sigo:cpu":
							// value must follow
							if count == 2 {
								p.targetInfo["cpu"] = parts[1]
							} else {
								// TODO: Return syntax error
							}
						case "//sigo:triple":
							// value must follow
							if count == 2 {
								p.targetInfo["triple"] = parts[1]
							} else {
								// TODO: Return syntax error
							}
						case "//sigo:extern":
							if count == 3 {
								symbolName := types.Id(pkg.Types, parts[1])
								info := p.options.GetSymbolInfo(symbolName)
								info.LinkName = parts[2]
								info.ExternalLinkage = true
							} else {
								// TODO: Return syntax error
							}
						//***********************************************
						case "//go:linkname":
							if count == 3 {
								symbolName := types.Id(pkg.Types, parts[1])
								info := p.options.GetSymbolInfo(symbolName)

								// NOTE: Allow multiple functions to use the same linkname. The compiler will assert
								//       that there is only one definition of it
								info.LinkName = parts[2]
							} else {
								// TODO: Return syntax error
							}
						case "//go:export":
							if count == 3 {
								funcName := types.Id(pkg.Types, parts[1])
								info := p.options.GetSymbolInfo(funcName)
								info.LinkName = parts[2]
								info.Exported = true
							} else {
								// TODO: Return syntax error
							}
						}
					}
				}
			}
		}
	}

	return
}

func (p *Program) buildPackages() ([]*ssa.Package, error) {
	mode := ssa.SanityCheckFunctions | ssa.BareInits | ssa.GlobalDebug | ssa.InstantiateGenerics | ssa.NaiveForm
	p.ssaProg = ssa.NewProgram(p.fset, mode)

	// Create all the package
	for _, pkg := range p.pkgs {
		// Import the package
		p.ssaProg.CreatePackage(pkg.Types, pkg.Syntax, pkg.TypesInfo, true)
	}

	// Build the SSA
	p.ssaProg.Build()

	return p.ssaProg.AllPackages(), nil
}

func (p *Program) nearestFuncBelowComment(comment *ast.Comment, file *ast.File) (fn *ast.FuncDecl) {
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.FuncDecl:
			// The comment must be above the function declaration.
			if comment.Pos() < n.Pos() {
				// Is this the top-most function declaration relative to below
				// the comment?
				if fn == nil || n.Pos() < fn.Pos() {
					fn = n
				}
			}
		}
		// TODO: Should probably consider other non-function nodes in between
		//       the found function and the comment.
		return true
	})

	return
}

func gatherPackages(pkg *packages.Package, in map[string]*packages.Package) map[string]*packages.Package {
	for _, imported := range pkg.Imports {
		in = gatherPackages(imported, in)
		in[imported.PkgPath] = imported
	}
	in[pkg.PkgPath] = pkg
	return in
}
