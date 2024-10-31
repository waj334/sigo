package ssa

import (
	"context"
	"errors"
	"go/ast"
	"go/importer"
	"go/token"
	"go/types"
	"gonum.org/v1/gonum/graph/multi"
	"gonum.org/v1/gonum/graph/topo"
	"hash/fnv"
	"io/fs"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/packages"
)

type ProgramConfig struct {
	Tags               []string
	AdditionalPackages []string
	Environment        []string
	PackagePath        string
	GoRoot             string
	Sizes              types.Sizes
}

type Program struct {
	Packages        map[string]*packages.Package
	OrderedPackages []*packages.Package
	Types           map[*packages.Package]map[string]types.Type
	Defines         map[string]string
	Files           map[string][]string
	LinkerScript    string
	FileSet         *token.FileSet
	Info            *types.Info
	Symbols         *SymbolInfoStore
	Config          *ProgramConfig
	MainFunc        string
	PackageInits    []*ast.Ident

	packageNodes    map[*packages.Package]*packageNode
	defaultImporter types.Importer
}

func (p *Program) Import(path string) (*types.Package, error) {
	if pkg, ok := p.Packages[path]; ok {
		return pkg.Types, nil
	}
	return p.defaultImporter.Import(path)
}

func (p *Program) ImportFrom(path, dir string, mode types.ImportMode) (*types.Package, error) {
	return p.Import(path)
}

type packageNode struct {
	pkg *packages.Package
	id  int64
}

func (p *packageNode) ID() int64 {
	return p.id
}

func NewProgram(config *ProgramConfig) *Program {
	return &Program{
		Packages:        map[string]*packages.Package{},
		OrderedPackages: []*packages.Package{},
		Types:           map[*packages.Package]map[string]types.Type{},
		Defines:         map[string]string{},
		Files:           map[string][]string{},
		FileSet:         token.NewFileSet(),
		Symbols:         NewSymbolInfoStore(),
		Config:          config,
		packageNodes:    map[*packages.Package]*packageNode{},
		defaultImporter: importer.Default(),
	}
}

func (p *Program) makeNode(pkg *packages.Package) *packageNode {
	// Look up an existing node for this package.
	if node, ok := p.packageNodes[pkg]; ok {
		return node
	}

	// Make a new node for this package.
	hasher := fnv.New64()
	hasher.Write([]byte(pkg.PkgPath))
	return &packageNode{
		pkg: pkg,
		id:  int64(hasher.Sum64()),
	}
}

func (p *Program) Parse(ctx context.Context) error {
	// Create the parser configuration.
	parserConfig := packages.Config{
		Mode:    packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedModule | packages.NeedEmbedFiles | packages.NeedEmbedPatterns | packages.NeedCompiledGoFiles,
		Context: ctx,
		Logf:    nil,
		Dir:     "",
		Env:     p.Config.Environment,
		BuildFlags: []string{
			"-tags=" + strings.Join(p.Config.Tags, ","),
		},
		Fset:    p.FileSet,
		Tests:   false,
		Overlay: nil,
	}

	// Collect the packages to be parsed.
	packagePaths := []string{"runtime", p.Config.PackagePath}
	packagePaths = append(packagePaths, p.Config.AdditionalPackages...)

	// Parse the packages.
	pkgs, err := packages.Load(&parserConfig, packagePaths...)
	if err != nil {
		return err
	}

	// Add all parse packages (including their imported packages).
	for _, pkg := range pkgs {
		// Add the package to the program.
		if pkgErr := p.AddPackage(pkg); pkgErr != nil {
			err = errors.Join(err, pkgErr)
		}
	}

	// Return early with error.
	if err != nil {
		return err
	}

	// Compute dependency graph.
	sortErr := p.computePackageOrder()
	if sortErr != nil {
		return errors.Join(err, sortErr)
	}

	// Locate linker script.
	linkerScripts := append(p.Files[".ld"], p.Files[".linker"]...)
	for _, fname := range linkerScripts {
		// TODO: Make script in main package directory take priority.
		if strings.Split(filepath.Base(fname), ".")[0] == "target" {
			p.LinkerScript = fname
		}
	}

	// Locate the main function symbol.
	for _, pkg := range p.Packages {
		if pkg.Module != nil && pkg.Module.Main {
			for _, file := range pkg.Syntax {
				for _, decl := range file.Decls {
					if decl, ok := decl.(*ast.FuncDecl); ok {
						if decl.Name.Name == "main" {
							// Set the main function symbol. This symbol will be mapped to "main.main" during linking.
							p.MainFunc = mangleSymbol(qualifiedName("main", pkg.Types))

							// Stop searching.
							goto done
						}
					}
				}
			}
		}
	}

done:
	return err
}

func (p *Program) computePackageOrder() error {
	// Create a directed graph that will be used to sort the packaged topologically in order of dependency.
	graph := multi.NewDirectedGraph()
	runtimePackageNode := p.makeNode(p.Packages["runtime"])
	for _, pkg := range p.Packages {
		// Add graph edges.
		pkgNode := p.makeNode(pkg)

		// Exclude the runtime package.
		// TODO: This might be problematic if any package runtime is dependent on has package initializers.
		if pkg.PkgPath != "runtime" {
			// All other packages implicitly depend on the runtime package.
			graph.SetLine(graph.NewLine(runtimePackageNode, pkgNode))

			// Add edges to imported packages.
			for _, imported := range pkg.Imports {
				importedPkgNode := p.makeNode(imported)
				graph.SetLine(graph.NewLine(importedPkgNode, pkgNode))
			}
		}
	}

	sorted, sortErr := topo.Sort(graph)
	if sortErr != nil {
		return sortErr
	}

	p.OrderedPackages = make([]*packages.Package, len(sorted))
	for i, node := range sorted {
		p.OrderedPackages[i] = node.(*packageNode).pkg
	}

	return nil
}

func (p *Program) AddPackage(pkg *packages.Package) (err error) {
	if _, ok := p.Packages[pkg.PkgPath]; ok {
		// Do not process this package again.
		return nil
	}

	defer func() {
		// Update package mappings.
		p.Packages[pkg.PkgPath] = pkg
	}()

	// Fail early by returning errors (if any).
	if len(pkg.Errors) > 0 {
		for _, pkgErr := range pkg.Errors {
			pos := strings.Split(pkgErr.Pos, ":")
			if strings.Index(pkgErr.Pos, ":") == 1 {
				// This is a Windoze path. Merge the first 2 elements.
				newPos := []string{pos[0] + ":" + pos[1]}
				if len(pos) > 2 {
					pos = append(newPos, pos[2:]...)
				} else {
					pos = newPos
				}
			}

			for i := 0; i < len(pos); i++ {
				evalPkgDir, symlinkErr := filepath.EvalSymlinks(pos[i])
				if symlinkErr == nil {
					pos[i] = evalPkgDir
				}
			}

			pkgErr.Pos = strings.Join(pos, ":")
			err = errors.Join(err, pkgErr)
		}
		return err
	}

	// Locate this package on the filesystem.
	pkgDir := pkg.PkgPath
	if pkg.Module == nil {
		// Assume this is a runtime package.
		// TODO: Support GOPATH?
		pkgDir = filepath.Join(p.Config.GoRoot, "src", pkgDir)
	} else {
		// Prepend the module directory to the package path.
		pkgDir = filepath.Join(pkg.Module.Dir, pkgDir)
	}

	// Evaluate symbolic links.
	evalPkgDir, symlinkErr := filepath.EvalSymlinks(pkgDir)
	if symlinkErr == nil {
		pkgDir = evalPkgDir
	}

	// Walk this package directory for files.
	filepath.WalkDir(pkgDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() && path != pkgDir {
			return fs.SkipDir
		}

		if !d.IsDir() {
			ext := strings.ToLower(filepath.Ext(path))
			// Insert into respective fileset.
			files := p.Files[ext]
			p.Files[ext] = append(files, path)
		}

		return nil
	})

	// Create type mappings.
	p.Types[pkg] = map[string]types.Type{}

	// Collect various information from this package.
	for _, file := range pkg.Syntax {
		// Parse all comments for pragma
		p.parsePragmas(file, pkg.Types)

		// Collect all declared types in this package.
		for _, decl := range file.Decls {
			switch decl := decl.(type) {
			case *ast.GenDecl:
				switch decl.Tok {
				case token.TYPE:
					for _, spec := range decl.Specs {
						id := spec.(*ast.TypeSpec).Name
						p.Types[pkg][id.Name] = pkg.TypesInfo.Defs[id].Type()
					}
				}
			}
		}
	}

	// Add any imported package.
	for _, imported := range pkg.Imports {
		pkgErr := p.AddPackage(imported)
		if pkgErr != nil {
			err = errors.Join(err, pkgErr)
		}
	}

	return err
}

func (p *Program) LookupType(pkgname, typename string) types.Type {
	if pkg, ok := p.Packages[pkgname]; ok {
		if _types, ok := p.Types[pkg]; ok {
			if T, ok := _types[typename]; ok {
				return T
			}
		}
	}
	return nil
}

func (p *Program) parsePragmas(file *ast.File, pkg *types.Package) {
	for _, commentGroup := range file.Comments {
		for _, comment := range commentGroup.List {
			// Split the comment on the space character
			parts := strings.Split(comment.Text, " ")
			if count := len(parts); count > 1 {
				// Process the comment based of the first part
				switch parts[0] {
				case "//sigo:extern":
					if count == 3 {
						_symbolName := mangleSymbol(qualifiedName(parts[1], pkg))
						info := p.Symbols.GetSymbolInfo(_symbolName)
						info.LinkName = mangleSymbol(parts[2])
						info.ExternalLinkage = true
					} else {
						// TODO: Return syntax error
					}
				case "//sigo:interrupt":
					if count == 3 {
						funcName := mangleSymbol(qualifiedName(parts[1], pkg))
						info := p.Symbols.GetSymbolInfo(funcName)
						info.LinkName = mangleSymbol(parts[2])
						info.IsInterrupt = true
						info.Exported = true
					} else {
						// TODO: Return syntax error
					}
				case "//sigo:define":
					if count == 2 {
						p.Defines[parts[1]] = ""
					} else {
						p.Defines[parts[1]] = parts[2]
					}
				case "//go:linkname", "//sigo:linkname":
					if count == 3 {
						_symbolName := mangleSymbol(qualifiedName(parts[1], pkg))
						info := p.Symbols.GetSymbolInfo(_symbolName)

						// NOTE: Allow multiple functions to use the same linkname. The compiler will assert
						//       that there is only one definition of it
						info.LinkName = mangleSymbol(parts[2])
					} else {
						// TODO: Return syntax error
					}
				case "//go:export", "//sigo:export":
					if count == 3 {
						funcName := mangleSymbol(qualifiedName(parts[1], pkg))
						info := p.Symbols.GetSymbolInfo(funcName)
						info.LinkName = mangleSymbol(parts[2])
						info.Exported = true
					} else {
						// TODO: Return syntax error
					}
				case "//sigo:linkage":
					if count == 3 {
						funcName := mangleSymbol(qualifiedName(parts[1], pkg))
						info := p.Symbols.GetSymbolInfo(funcName)
						info.Linkage = strings.ToLower(parts[2])
					}
				case "//sigo:required":
					funcName := mangleSymbol(qualifiedName(parts[1], pkg))
					info := p.Symbols.GetSymbolInfo(funcName)
					info.IsRequired = true
				}
			}
		}
	}
}
