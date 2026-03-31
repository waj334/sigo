package ssa

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"hash/fnv"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"golang.org/x/tools/go/packages"
	"gonum.org/v1/gonum/graph/multi"
	"gonum.org/v1/gonum/graph/topo"

	"pkg.si-go.dev/sigo/compiler/check"
)

var pragmaRegex = regexp.MustCompile(`^//[\t\f\v ]*(?:go|sigo):[\t\f\v ]*([a-zA-Z0-9 ./_]+)$`)
var embedRegex = regexp.MustCompile(`^//[\t\f\v ]*go:embed[\t\f\v ]+(.+)$`)

type ProgramConfig struct {
	Tags               []string
	AdditionalPackages []string
	Environment        []string
	PackagePath        string
	ModuleRoot         string
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
	EmbedContents   map[string][]byte // keyed by qualified symbol name
	CGoPreambles    []string          // C preambles extracted from import "C" doc comments

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
		EmbedContents:   map[string][]byte{},
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
	// Pre-scan for import "C" files: extract preambles and build overlays
	// that strip the CGo import so the Go type-checker never invokes cgo.
	overlay, err := p.scanCGoFiles()
	if err != nil {
		return err
	}

	// Create the parser configuration.
	parserConfig := packages.Config{
		Mode:    packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedDeps | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedModule | packages.NeedEmbedFiles | packages.NeedEmbedPatterns,
		Context: ctx,
		Logf:    nil,
		Dir:     p.Config.ModuleRoot,
		Env:     p.Config.Environment,
		BuildFlags: []string{
			"-tags=" + strings.Join(p.Config.Tags, ","),
		},
		Fset:    p.FileSet,
		Tests:   false,
		Overlay: overlay,
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

	// Perform additional type checking required by SiGo.
	var checkErr error
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			err := check.CheckAST(pkg.Fset, pkg.Types, file, pkg.TypesInfo)
			if err != nil {
				checkErr = errors.Join(checkErr, err)
			}
		}
	}

	if checkErr != nil {
		return checkErr
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
		baseName := strings.TrimSuffix(filepath.Base(fname), filepath.Ext(fname))
		if strings.Contains(baseName, "target") || strings.Contains(baseName, "linker") {
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
							if decl.Body == nil {
								// This is likely a forward declaration of the main function for use in a runtime
								// implementation. Skip it.
								continue
							}
							// Set the main function symbol. This symbol will be mapped to "main.main" during linking.
							p.MainFunc = qualifiedName("main", pkg.Types)

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
				if imported.PkgPath == "C" {
					continue
				}
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

	// Skip the CGo pseudo-package — sigo handles C interop via its own CIR pipeline.
	if pkg.PkgPath == "C" {
		return nil
	}

	defer func() {
		// Update package mappings.
		p.Packages[pkg.PkgPath] = pkg
	}()

	// Fail early by returning errors (if any).
	if len(pkg.Errors) > 0 {
		for _, pkgErr := range pkg.Errors {
			// Skip CGo errors — sigo resolves import "C" via its own CIR pipeline,
			// so the Go type-checker not finding the "C" package is expected.
			if strings.Contains(pkgErr.Msg, "could not import C") ||
				strings.Contains(pkgErr.Msg, "no metadata for C") {
				continue
			}
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
		// Only return early if there are non-CGo errors.
		if err != nil {
			return err
		}
	}

	// Locate this package on the filesystem.
	pkgDir := pkg.Dir

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
			fname := filepath.Base(path)
			fname = strings.TrimSuffix(fname, filepath.Ext(fname))
			tag := strings.Split(fname, "_")
			if len(tag) > 1 {
				if !slices.Contains(p.Config.Tags, tag[len(tag)-1]) {
					// Stop processing this file
					return nil
				}
			}

			// Insert into respective fileset.
			ext := strings.ToLower(filepath.Ext(path))
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
			if decl, ok := decl.(*ast.GenDecl); ok && decl.Tok == token.TYPE {
				for _, spec := range decl.Specs {
					id := spec.(*ast.TypeSpec).Name
					p.Types[pkg][id.Name] = pkg.TypesInfo.Defs[id].Type()
				}
			}
		}
	}

	// Resolve any //go:embed directives in this package.
	if embedErr := p.resolveEmbedData(pkg); embedErr != nil {
		err = errors.Join(err, embedErr)
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

func (p *Program) resolveEmbedData(pkg *packages.Package) error {
	// Build a set of files that go/packages says are embeddable for this package.
	if len(pkg.EmbedFiles) == 0 {
		return nil
	}

	// Iterate all symbols looking for those with embed patterns in this package.
	p.Symbols.mu.Lock()
	defer p.Symbols.mu.Unlock()

	prefix := pkg.PkgPath + "."
	for symbol, info := range p.Symbols.info {
		if len(info.EmbedPatterns) == 0 {
			continue
		}
		if !strings.HasPrefix(symbol, prefix) {
			continue
		}

		// Match patterns against embeddable files.
		var matched []string
		for _, pattern := range info.EmbedPatterns {
			for _, f := range pkg.EmbedFiles {
				// Match against the relative path from the package directory.
				rel, err := filepath.Rel(pkg.Dir, f)
				if err != nil {
					continue
				}
				// Use forward slashes for matching (Go embed uses forward slashes).
				rel = filepath.ToSlash(rel)
				if ok, _ := path.Match(pattern, rel); ok {
					matched = append(matched, f)
				}
			}
		}

		if len(matched) == 0 {
			return fmt.Errorf("//go:embed: pattern %v matches no files for %s", info.EmbedPatterns, symbol)
		}

		if len(matched) > 1 {
			return fmt.Errorf("//go:embed: patterns match multiple files for %s (string and []byte require exactly one file)", symbol)
		}

		// Read the file content.
		content, err := os.ReadFile(matched[0])
		if err != nil {
			return fmt.Errorf("//go:embed: cannot read %s: %w", matched[0], err)
		}

		p.EmbedContents[symbol] = content
	}

	return nil
}

func (p *Program) parsePragmas(file *ast.File, pkg *types.Package) {
	// Build a map of declaration positions to their declarations for quick lookup
	declsByPos := make(map[token.Pos]ast.Decl)
	for _, decl := range file.Decls {
		declsByPos[decl.Pos()] = decl
	}

	// Process each comment group
	for _, commentGroup := range file.Comments {
		// Find the declaration immediately following this comment group
		var targetDecl ast.Decl
		var targetDeclPos token.Pos = token.NoPos

		// Find the nearest declaration after this comment
		for pos, decl := range declsByPos {
			if pos > commentGroup.End() {
				if targetDeclPos == token.NoPos || pos < targetDeclPos {
					targetDeclPos = pos
					targetDecl = decl
				}
			}
		}

		// Extract symbol name from the target declaration if found
		var symbolName string
		if targetDecl != nil {
			switch decl := targetDecl.(type) {
			case *ast.FuncDecl:
				// Function declaration
				if decl.Recv != nil && len(decl.Recv.List) > 0 {
					// Method: extract receiver type
					recvType := decl.Recv.List[0].Type
					// Handle pointer receivers
					if star, ok := recvType.(*ast.StarExpr); ok {
						recvType = star.X
					}
					if ident, ok := recvType.(*ast.Ident); ok {
						symbolName = qualifiedName(ident.Name+"."+decl.Name.Name, pkg)
					}
				} else {
					// Regular function
					symbolName = qualifiedName(decl.Name.Name, pkg)
				}
			case *ast.GenDecl:
				// Variable, constant, or type declaration
				if len(decl.Specs) > 0 {
					switch spec := decl.Specs[0].(type) {
					case *ast.ValueSpec:
						// Variable or constant
						if len(spec.Names) > 0 {
							symbolName = qualifiedName(spec.Names[0].Name, pkg)
						}
					case *ast.TypeSpec:
						// Type declaration
						symbolName = qualifiedName(spec.Name.Name, pkg)
					}
				}
			}
		}

		// Process each comment in the group
		for _, comment := range commentGroup.List {
			// Check for //go:embed directive first (separate regex due to glob chars)
			if embedMatches := embedRegex.FindStringSubmatch(comment.Text); len(embedMatches) > 0 {
				if symbolName != "" {
					info := p.Symbols.GetSymbolInfo(symbolName)
					// Split on whitespace to support multiple patterns on one line
					patterns := strings.Fields(embedMatches[1])
					info.EmbedPatterns = append(info.EmbedPatterns, patterns...)
				}
				continue
			}

			matches := pragmaRegex.FindStringSubmatch(comment.Text)
			if len(matches) == 0 {
				continue
			}

			// Split the arguments on the space character
			parts := strings.Fields(matches[1])
			if len(parts) == 0 {
				continue
			}

			// Determine which symbol this pragma applies to
			var targetSymbol string
			count := len(parts)

			// Check if the pragma explicitly names a symbol (old style)
			if count > 1 {
				// Some pragmas like "extern", "export", etc. include the symbol name
				// These still work with explicit symbol names for backwards compatibility
				switch parts[0] {
				case "extern", "interrupt", "linkname", "export", "linkage", "required", "section":
					// These pragmas include the symbol name as the second argument.
					if count >= 2 {
						targetSymbol = qualifiedName(parts[1], pkg)
					}
				}
			}

			// If no explicit symbol, use the symbol from the following declaration
			if targetSymbol == "" && symbolName != "" {
				targetSymbol = symbolName
			}

			// Process the pragma
			switch parts[0] {
			case "extern":
				if count == 3 {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[2]
					info.ExternalLinkage = true
				} else if count == 2 && targetSymbol != "" {
					// New style: //go:extern linkname
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[1]
					info.ExternalLinkage = true
				}
			case "interrupt":
				if count == 3 {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[2]
					info.IsInterrupt = true
					info.Exported = true
					info.Attributes["nowritebarrier"] = struct{}{}
				} else if count == 2 && targetSymbol != "" {
					// New style: //go:interrupt linkname
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[1]
					info.IsInterrupt = true
					info.Exported = true
					info.Attributes["nowritebarrier"] = struct{}{}
				}
			case "define":
				if count == 2 {
					p.Defines[parts[1]] = ""
				} else if count >= 3 {
					p.Defines[parts[1]] = strings.Join(parts[2:], " ")
				}
			case "linkname":
				if count == 3 {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[2]
				} else if count == 2 && targetSymbol != "" {
					// New style: //go:linkname externalname
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[1]
				}
			case "export":
				if count >= 2 {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Exported = true
					if count == 3 {
						info.LinkName = parts[2]
					} else if count == 2 && targetSymbol != symbolName {
						// Old style with explicit symbol name
						info.LinkName = parts[1]
					}
				} else if count == 1 && targetSymbol != "" {
					// New style: //go:export (uses symbol name as linkname)
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Exported = true
				}
			case "linkage":
				if count == 3 {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Linkage = strings.ToLower(parts[2])
				} else if count == 2 && targetSymbol != "" {
					// New style: //go:linkage weak
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Linkage = strings.ToLower(parts[1])
				}
			case "required":
				if targetSymbol != "" {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.IsRequired = true
				} else if count >= 2 {
					// Old style with explicit symbol name
					funcName := qualifiedName(parts[1], pkg)
					info := p.Symbols.GetSymbolInfo(funcName)
					info.IsRequired = true
				}
			case "nosplit", "nowritebarrier":
				if targetSymbol != "" {
					// New style: //go:nosplit or //go:nowritebarrier
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Attributes[parts[0]] = struct{}{}
				} else if count > 2 {
					// Old style with explicit symbol name and attributes
					funcName := qualifiedName(parts[1], pkg)
					info := p.Symbols.GetSymbolInfo(funcName)
					info.Attributes[parts[2]] = struct{}{}
				}
			case "section":
				if count >= 2 && targetSymbol != "" {
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.Section = parts[1]
				} else if count > 2 {
					// Old style with explicit symbol name
					funcName := qualifiedName(parts[1], pkg)
					info := p.Symbols.GetSymbolInfo(funcName)
					info.Section = parts[2]
				}
			}
		}
	}
}

// scanCGoFiles walks the package directory looking for Go files that contain
// import "C". For each such file it:
//   - extracts the C preamble from the doc comment and appends it to CGoPreambles
//   - produces an overlay entry with the import "C" declaration (and its preceding
//     doc comment) blanked out, so the Go type-checker never attempts CGo processing
func (p *Program) scanCGoFiles() (map[string][]byte, error) {
	overlay := map[string][]byte{}
	err := filepath.WalkDir(p.Config.PackagePath, func(fpath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() || !strings.HasSuffix(fpath, ".go") {
			return walkErr
		}
		src, readErr := os.ReadFile(fpath)
		if readErr != nil {
			return readErr
		}
		if !bytes.Contains(src, []byte(`"C"`)) {
			return nil
		}

		fset := token.NewFileSet()
		f, parseErr := parser.ParseFile(fset, fpath, src, parser.ParseComments)
		if parseErr != nil {
			return nil // let packages.Load surface the real parse error
		}

		result := make([]byte, len(src))
		copy(result, src)
		modified := false

		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.IMPORT {
				continue
			}
			for _, spec := range gd.Specs {
				is, ok := spec.(*ast.ImportSpec)
				if !ok || is.Path.Value != `"C"` {
					continue
				}
				modified = true

				// Extract the preamble from the preceding doc comment.
				if gd.Doc != nil {
					p.CGoPreambles = append(p.CGoPreambles, gd.Doc.Text())
				}

				// Determine the byte range to blank out.
				var start, end int
				if len(gd.Specs) == 1 {
					// The entire GenDecl is just import "C"; remove it
					// along with its doc comment (the C preamble).
					if gd.Doc != nil {
						start = fset.Position(gd.Doc.Pos()).Offset
					} else {
						start = fset.Position(gd.Pos()).Offset
					}
					end = fset.Position(gd.End()).Offset
				} else {
					// Multi-import block: only blank the "C" spec line.
					start = fset.Position(is.Pos()).Offset
					end = fset.Position(is.End()).Offset
				}

				// Replace with spaces, preserving newlines so line numbers stay intact.
				for i := start; i < end && i < len(result); i++ {
					if result[i] != '\n' {
						result[i] = ' '
					}
				}
			}
		}

		if modified {
			overlay[fpath] = result
		}
		return nil
	})
	return overlay, err
}
