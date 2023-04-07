package builder

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
	"path/filepath"
	"strings"
)

type Program struct {
	config  *types.Config
	files   []*ast.File
	fset    *token.FileSet
	mainPkg *types.Package
	path    string

	pkgs       map[string]*ssa.Package
	targetInfo map[string]string
}

func (p *Program) parse() (err error) {
	// Parse the package at the directory
	packages, err := parser.ParseDir(p.fset, p.path, nil, parser.ParseComments)
	if err != nil {
		return err
	}

	for name, pkg := range packages {
		// Create a package
		p.mainPkg = types.NewPackage(filepath.Base(p.path), name)

		// Gather files from packages and information from any special comments
		// found in each source.
		for _, file := range pkg.Files {
			p.files = append(p.files, file)

			for _, commentGroup := range file.Comments {
				for _, comment := range commentGroup.List {
					// Split the comment on the space character
					parts := strings.Split(comment.Text, " ")
					if count := len(parts); count > 1 {
						// Process the comment based of the first part
						switch parts[0] {
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
						}
					}
				}
			}
		}

		if len(packages) > 1 {
			panic("Refactor this")
		}
		break
	}

	return nil
}

func (p *Program) buildPackages() error {
	if len(p.pkgs) == 0 {
		mode := ssa.SanityCheckFunctions | ssa.BareInits | ssa.GlobalDebug | ssa.InstantiateGenerics
		info := &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Scopes:     make(map[ast.Node]*types.Scope),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
		}

		if err := types.NewChecker(p.config, p.fset, p.mainPkg, info).Files(p.files); err != nil {
			return err
		}

		prog := ssa.NewProgram(p.fset, mode)

		// Create SSA packages for all imports.
		// Order is not significant.
		created := make(map[*types.Package]bool)
		var createAll func(pkgs *types.Package)
		createAll = func(pkg *types.Package) {
			for _, imported := range pkg.Imports() {
				if !created[imported] {
					created[imported] = true
					importer := p.config.Importer.(*Importer)
					p.pkgs[imported.Path()] = prog.CreatePackage(imported, importer.files[imported], importer.info[imported], true)
					createAll(imported)
				}
			}
		}
		createAll(p.mainPkg)

		// Create and build the main package.
		ssapkg := prog.CreatePackage(p.mainPkg, p.files, info, false)
		prog.Build()
		p.pkgs[p.mainPkg.Path()] = ssapkg
	}
	return nil
}
