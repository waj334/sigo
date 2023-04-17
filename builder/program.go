package builder

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"os"
	"path/filepath"
	"strings"
)

type Program struct {
	config types.Config
	files  []*ast.File
	fset   *token.FileSet
	path   string
	env    Env

	prog       *loader.Program
	ssaProg    *ssa.Program
	targetInfo map[string]string
	linkNames  map[string]string
}

func (p *Program) parse() (err error) {
	/*// Parse the package at the directory
	packages, err := parser.ParseDir(p.fset, p.path, nil, parser.ParseComments)
	if err != nil {
		return err
	}*/

	// Create a context to load packages under
	ctx := build.Context{
		GOARCH:        "amd64",
		GOOS:          "linux",
		GOROOT:        p.env.Value("SIGOROOT"),
		GOPATH:        p.env.Value("SIGOPATH"),
		Dir:           "",
		CgoEnabled:    true,
		UseAllFiles:   false,
		Compiler:      "gc",
		BuildTags:     nil,
		ToolTags:      nil,
		ReleaseTags:   nil,
		InstallSuffix: "",
		JoinPath:      nil,
		SplitPathList: nil,
		IsAbsPath:     nil,
		IsDir:         nil,
		HasSubdir:     nil,
		ReadDir:       nil,
		OpenFile:      nil,
	}

	conf := loader.Config{
		Fset:                p.fset,
		ParserMode:          parser.ParseComments,
		TypeChecker:         p.config,
		TypeCheckFuncBodies: nil,
		Build:               nil,
		Cwd:                 "",
		DisplayPath:         nil,
		AllowErrors:         false,
		CreatePkgs:          nil,
		ImportPkgs: map[string]bool{
			"runtime/internal/go": true,
		},
		FindPackage: func(ctxt *build.Context, importPath, fromDir string, mode build.ImportMode) (*build.Package, error) {
			return ctx.Import(importPath, fromDir, mode)
		},
		AfterTypeCheck: nil,
	}

	// Import the required runtime packages first
	//conf.Import("runtime/internal/types")

	// Import the main program package
	conf.Import(p.path)

	// Load all the packages and their dependencies
	p.prog, err = conf.Load()
	if err != nil {
		return err
	}

	for _, pkg := range p.prog.AllPackages {
		// Get this package's imports so required packages can be appended to it
		imports := pkg.Pkg.Imports()

		if strings.Index(pkg.Pkg.Path(), "runtime/internal/") != 0 {
			// Append the runtime package to the imports
			imports = append(imports, p.prog.Package("runtime/internal/go").Pkg)
		}

		// Replace the package's imports with the new list
		pkg.Pkg.SetImports(imports)

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
						//***********************************************
						case "//go:linkname":
							if count == 2 {
								// Which function does this comment affect?
								funcDecl := p.nearestFuncBelowComment(comment, file)
								if funcDecl != nil {
									name := fmt.Sprint(pkg.Pkg.Path(), ".", funcDecl.Name.Name)

									// NOTE: Allow multiple functions to use the same linkname. The compiler will assert
									//       that there is only one definition of it
									p.linkNames[name] = parts[1]
								}
							} else {
								// TODO: Return syntax error
							}
						}
					}
				}
			}
		}
	}

	return nil
}

func (p *Program) importPackage(path string, srcDir string, mode build.ImportMode) (*build.Package, error) {
	searchPaths := []string{
		filepath.Clean(fmt.Sprintf("%s/%s", p.path, path)),
		filepath.Clean(fmt.Sprintf("%s/src/%s", p.env.Value("SIGOROOT"), path)),
		// TODO: Resolve package directories in GOMODPATH
	}

	var lookupPath string
	for _, searchPath := range searchPaths {
		// Attempt to stat the directory
		info, err := os.Stat(searchPath)
		if os.IsNotExist(err) || !info.IsDir() {
			continue
		}
		lookupPath = searchPath
		break
	}

	if len(lookupPath) == 0 {
		return nil, os.ErrNotExist
	}

	pkg := build.Package{
		ImportPath: path,
	}

	return &pkg, nil
}

func (p *Program) buildPackages() ([]*ssa.Package, error) {
	mode := ssa.SanityCheckFunctions | ssa.BareInits | ssa.GlobalDebug | ssa.InstantiateGenerics
	p.ssaProg = ssa.NewProgram(p.fset, mode)

	// Create all the package
	for _, pkg := range p.prog.AllPackages {
		p.ssaProg.CreatePackage(pkg.Pkg, pkg.Files, &pkg.Info, true)
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
