package builder

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
)

type Importer struct {
	env         Env
	imported    map[string]*types.Package
	files       map[*types.Package][]*ast.File
	info        map[*types.Package]*types.Info
	fset        *token.FileSet
	buildPkgDir string
}

func (i *Importer) Import(path string) (*types.Package, error) {
	// Check the map of the already imported packages first
	if pkg, ok := i.imported[path]; ok {
		return pkg, nil
	}

	searchPaths := []string{
		filepath.Clean(fmt.Sprintf("%s/%s", i.buildPkgDir, path)),
		filepath.Clean(fmt.Sprintf("%s/src/%s", i.env.Value("SIGOROOT"), path)),
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

	// Found a directory matching the path. Parse the package
	packages, err := parser.ParseDir(i.fset, lookupPath, nil, 0)

	// Get the package from the map
	var pkg *types.Package
	if len(packages) > 1 {
		return nil, ErrMultiplePackages
	} else {
		for _, node := range packages {
			conf := types.Config{Importer: i}
			info := &types.Info{
				Types:      make(map[ast.Expr]types.TypeAndValue),
				Defs:       make(map[*ast.Ident]types.Object),
				Uses:       make(map[*ast.Ident]types.Object),
				Implicits:  make(map[ast.Node]types.Object),
				Scopes:     make(map[ast.Node]*types.Scope),
				Selections: make(map[*ast.SelectorExpr]*types.Selection),
			}

			var files []*ast.File
			for _, file := range node.Files {
				files = append(files, file)
			}

			pkg, err = conf.Check(path, i.fset, files, info)
			if err != nil {
				return nil, err
			}

			// Cache the files
			i.files[pkg] = files
			i.info[pkg] = info

			// Cache the package
			i.imported[path] = pkg
			break
		}
	}

	return pkg, nil
}
