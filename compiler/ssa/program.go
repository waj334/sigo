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
	"sort"
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
	DependencyDirs     map[string]string // pkgPath -> absolute directory (from first packages.Load)
	TargetTriplet      string            // target triple for C preprocessing
	IncludePaths       []string          // system include paths for C preprocessing
}

// cgoFlagRegex matches a #cgo pragma line in a C preamble comment.
// Group 1: flag name (CFLAGS, CPPFLAGS, CXXFLAGS, FFLAGS, LDFLAGS)
// Group 2: flag arguments
var cgoFlagRegex = regexp.MustCompile(`(?m)^\s*#cgo\s+(CFLAGS|CPPFLAGS|CXXFLAGS|FFLAGS|LDFLAGS):\s*(.*)$`)

// CGoPreamble holds a C preamble extracted from an import "C" doc comment
// together with the Go source location where it was defined.
type CGoPreamble struct {
	Text    string   // C source text with #cgo lines stripped
	GoFile  string   // Go source file that contained the preamble
	GoLine  int      // Line number in the Go file where the preamble text starts
	CFlags  []string // flags from #cgo CFLAGS / CPPFLAGS / CXXFLAGS / FFLAGS
	LDFlags []string // flags from #cgo LDFLAGS
}

// parseCGoPragmas strips #cgo pragma lines from raw C preamble text,
// returning the cleaned text and the collected flags.
func parseCGoPragmas(text string) (cleaned string, cflags []string, ldflags []string) {
	var sb strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if m := cgoFlagRegex.FindStringSubmatch(line); m != nil {
			args := strings.Fields(m[2])
			switch m[1] {
			case "CFLAGS", "CPPFLAGS", "CXXFLAGS", "FFLAGS":
				cflags = append(cflags, args...)
			case "LDFLAGS":
				ldflags = append(ldflags, args...)
			}
			// Replace the line with blank space to preserve line numbers.
			sb.WriteByte('\n')
		} else {
			sb.WriteString(line)
			sb.WriteByte('\n')
		}
	}
	// Trim the extra trailing newline added by the loop.
	cleaned = strings.TrimSuffix(sb.String(), "\n")
	return
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
	EmbedContents   map[string][]byte      // keyed by qualified symbol name
	CGoPreambles    []CGoPreamble          // C preambles extracted from import "C" doc comments
	CGOFuncNames    map[string]bool        // C function names referenced via C.xxx (populated by scanCGoFiles)
	CGOLDFlags      []string               // linker flags from #cgo LDFLAGS directives
	CGOIncludePaths []string               // extra include paths from #cgo CFLAGS: -I...
	CGODefines      []string               // extra defines from #cgo CFLAGS: -D... (NAME or NAME=VALUE)
	CGOStaticFuncs  map[string]ASTFuncDecl // static C functions needing __sigo_wrap_ wrappers

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
		CGOFuncNames:    map[string]bool{},
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
					if p.CGOFuncNames[parts[2]] {
						info.IsCGoFunc = true
					}
				} else if count == 2 && targetSymbol != "" {
					// New style: //go:extern linkname
					info := p.Symbols.GetSymbolInfo(targetSymbol)
					info.LinkName = parts[1]
					info.ExternalLinkage = true
					if p.CGOFuncNames[parts[1]] {
						info.IsCGoFunc = true
					}
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
//   - blanks out the import "C" declaration (and its preceding doc comment)
//   - rewrites every C.Name selector expression to _cgo_Name
//
// After processing all files, it injects a synthetic _cgo_sigo_generated.go
// overlay file containing type aliases and function stubs for the referenced C
// symbols, so the Go type-checker can resolve them without invoking cgo.
func (p *Program) scanCGoFiles() (map[string][]byte, error) {
	overlay := map[string][]byte{}

	// Collect directories to scan: the main package and all dependencies.
	// Deduplicate so the main package isn't scanned twice (it also appears
	// in DependencyDirs under its real package path).
	dirsToScan := make(map[string]string) // label -> directory
	dirsToScan["main"] = p.Config.PackagePath
	mainAbs, _ := filepath.Abs(p.Config.PackagePath)
	for pkgPath, dir := range p.Config.DependencyDirs {
		abs, _ := filepath.Abs(dir)
		if abs == mainAbs {
			continue
		}
		dirsToScan[pkgPath] = dir
	}

	// Phase 1: scan all directories, collect preambles and CGo names.
	var scanResults []*cgoScanResult
	for _, dir := range dirsToScan {
		result, err := p.scanCGoDir(dir, overlay)
		if err != nil {
			return nil, err
		}
		if result == nil {
			continue
		}

		scanResults = append(scanResults, result)

		// Aggregate preambles and function names globally.
		p.CGoPreambles = append(p.CGoPreambles, result.preambles...)
		for name := range result.nameSet {
			p.CGOFuncNames[name] = true
		}
	}

	if len(scanResults) == 0 {
		return overlay, nil
	}

	// Aggregate #cgo compiler and linker flags from all preambles.
	for _, result := range scanResults {
		for _, pre := range result.preambles {
			for _, flag := range pre.CFlags {
				if strings.HasPrefix(flag, "-I") {
					p.CGOIncludePaths = append(p.CGOIncludePaths, flag[2:])
				} else if strings.HasPrefix(flag, "-D") {
					p.CGODefines = append(p.CGODefines, flag[2:])
				}
			}
			p.CGOLDFlags = append(p.CGOLDFlags, pre.LDFlags...)
		}
	}
	extraIncludePaths := p.CGOIncludePaths
	extraDefines := p.CGODefines

	// Phase 2: combine all preambles and extract typed function signatures
	// via Clang JSON AST.
	var combinedPreamble strings.Builder
	// Prepend -D defines as #define directives.
	for _, def := range extraDefines {
		eqIdx := strings.IndexByte(def, '=')
		if eqIdx >= 0 {
			fmt.Fprintf(&combinedPreamble, "#define %s %s\n", def[:eqIdx], def[eqIdx+1:])
		} else {
			fmt.Fprintf(&combinedPreamble, "#define %s\n", def)
		}
	}
	for _, pre := range p.CGoPreambles {
		if pre.GoFile != "" {
			fmt.Fprintf(&combinedPreamble, "#line %d \"%s\"\n", pre.GoLine, pre.GoFile)
		}
		combinedPreamble.WriteString(pre.Text)
		combinedPreamble.WriteByte('\n')
	}

	// Build a known-types map from preamble-local typedef structs for type
	// conversion of parameters/returns that reference them.
	knownTypes := make(map[string]string)
	for _, result := range scanResults {
		for _, pre := range result.preambles {
			for _, s := range parseCStructs(pre.Text) {
				knownTypes[s.name] = "_cgo_" + s.name
			}
		}
		// Add referenced type names as opaque types so that pointer parameters
		// like "struct pbuf *" resolve to "*_cgo_pbuf" instead of unsafe.Pointer.
		for name := range result.typeNames {
			if _, exists := knownTypes[name]; !exists {
				knownTypes[name] = "_cgo_" + name
			}
		}
	}

	var astFuncs []ASTFuncDecl
	if p.Config.TargetTriplet != "" {
		includePaths := append(p.Config.IncludePaths, extraIncludePaths...)
		jsonData, err := clangJSONAST(combinedPreamble.String(), p.Config.TargetTriplet, includePaths, nil)
		if err != nil {
			return nil, fmt.Errorf("clang AST dump failed: %w", err)
		}
		astFuncs, err = parseClangAST(jsonData, knownTypes)
		if err != nil {
			return nil, fmt.Errorf("parsing clang AST: %w", err)
		}
	}

	// Track static functions that need __sigo_wrap_ wrappers.
	p.CGOStaticFuncs = make(map[string]ASTFuncDecl)
	staticNames := make(map[string]bool)
	for _, fn := range astFuncs {
		if fn.IsStatic && p.CGOFuncNames[fn.Name] {
			p.CGOStaticFuncs[fn.Name] = fn
			staticNames[fn.Name] = true
		}
	}

	// Phase 3: generate per-package stub files from AST function signatures
	// and regex-parsed struct definitions from raw preamble text.
	for _, result := range scanResults {
		if len(result.cgoNames) > 0 && result.pkgName != "" {
			// Parse structs from raw preamble text (no preprocessing needed).
			var structs []cStructDecl
			for _, pre := range result.preambles {
				structs = append(structs, parseCStructs(pre.Text)...)
			}
			generated := generateCGoFileFromAST(result.pkgName, astFuncs, structs, result.cgoNames, result.typeNames, staticNames, knownTypes)
			genPath := filepath.Join(result.absDir, "cgo_sigo_generated.go")
			overlay[genPath] = []byte(generated)
		}
	}

	return overlay, nil
}

// cgoScanResult holds the results of scanning a single directory for CGo files.
type cgoScanResult struct {
	absDir    string
	pkgName   string
	cgoNames  []string
	nameSet   map[string]bool
	typeNames map[string]bool // C names used in type positions (type aliases, field types)
	preambles []CGoPreamble
}

// scanCGoDir scans a single directory for Go files that import "C". For each
// such file it strips the import, rewrites C.xxx to _cgo_xxx, and adds the
// modified source to overlay. Returns nil if no CGo files were found.
func (p *Program) scanCGoDir(dir string, overlay map[string][]byte) (*cgoScanResult, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}

	// Quick check: does this directory exist?
	if _, err := os.Stat(absDir); err != nil {
		return nil, nil
	}

	result := &cgoScanResult{
		absDir:    absDir,
		nameSet:   map[string]bool{},
		typeNames: map[string]bool{},
	}
	found := false

	err = filepath.WalkDir(absDir, func(fpath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		// Do not recurse into subdirectories — each package directory is
		// scanned independently via DependencyDirs.
		if d.IsDir() {
			if fpath != absDir {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(fpath, ".go") {
			return nil
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

		buf := make([]byte, len(src))
		copy(buf, src)
		modified := false

		// Record the package name for the generated file.
		if result.pkgName == "" && f.Name != nil {
			result.pkgName = f.Name.Name
		}

		// Step 1: blank out import "C" (and its C preamble doc comment).
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
					docPos := fset.Position(gd.Doc.Pos())
					goLine := docPos.Line
					if strings.HasPrefix(gd.Doc.List[0].Text, "/*") {
						goLine++
					}
					goFile := docPos.Filename
					if abs, err := filepath.Abs(goFile); err == nil {
						goFile = abs
					}
					cleaned, cflags, ldflags := parseCGoPragmas(gd.Doc.Text())
					result.preambles = append(result.preambles, CGoPreamble{
						Text:    cleaned,
						GoFile:  goFile,
						GoLine:  goLine,
						CFlags:  cflags,
						LDFlags: ldflags,
					})
				}

				// Determine the byte range to blank out.
				var start, end int
				if len(gd.Specs) == 1 {
					if gd.Doc != nil {
						start = fset.Position(gd.Doc.Pos()).Offset
					} else {
						start = fset.Position(gd.Pos()).Offset
					}
					end = fset.Position(gd.End()).Offset
				} else {
					start = fset.Position(is.Pos()).Offset
					end = fset.Position(is.End()).Offset
				}

				for i := start; i < end && i < len(buf); i++ {
					if buf[i] != '\n' {
						buf[i] = ' '
					}
				}
			}
		}

		if !modified {
			return nil
		}

		found = true

		// Step 2: collect all C.Name selector expressions and rewrite them to
		// _cgo_Name.
		type rewrite struct {
			start, end int
			name       string
		}
		var rewrites []rewrite
		ast.Inspect(f, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			id, ok := sel.X.(*ast.Ident)
			if !ok || id.Name != "C" {
				return true
			}
			rewrites = append(rewrites, rewrite{
				start: fset.Position(sel.Pos()).Offset,
				end:   fset.Position(sel.End()).Offset,
				name:  sel.Sel.Name,
			})
			return true
		})

		// Detect C names used in type positions so they can be excluded from
		// the __sigo_cgo_refs function-address array (taking &type is invalid C).
		isCSelector := func(expr ast.Expr) (string, bool) {
			if star, ok := expr.(*ast.StarExpr); ok {
				expr = star.X // unwrap pointer: *C.foo → C.foo
			}
			sel, ok := expr.(*ast.SelectorExpr)
			if !ok {
				return "", false
			}
			id, ok := sel.X.(*ast.Ident)
			if !ok || id.Name != "C" {
				return "", false
			}
			return sel.Sel.Name, true
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.TypeSpec:
				if name, ok := isCSelector(node.Type); ok {
					result.typeNames[name] = true
				}
			case *ast.Field:
				if name, ok := isCSelector(node.Type); ok {
					result.typeNames[name] = true
				}
			}
			return true
		})

		sort.Slice(rewrites, func(i, j int) bool {
			return rewrites[i].start > rewrites[j].start
		})

		for _, rw := range rewrites {
			replacement := []byte("_cgo_" + rw.name)
			buf = append(buf[:rw.start], append(replacement, buf[rw.end:]...)...)
			if !result.nameSet[rw.name] {
				result.nameSet[rw.name] = true
				result.cgoNames = append(result.cgoNames, rw.name)
			}
		}

		overlay[fpath] = buf
		return nil
	})
	if err != nil {
		return nil, err
	}

	if !found {
		return nil, nil
	}
	return result, nil
}
