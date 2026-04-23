package ssa

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// goKeywords is the set of Go reserved keywords. C parameter names that collide
// with these are prefixed with "_" in generated Go stubs.
var goKeywords = map[string]bool{
	"break": true, "case": true, "chan": true, "const": true, "continue": true,
	"default": true, "defer": true, "else": true, "fallthrough": true, "for": true,
	"func": true, "go": true, "goto": true, "if": true, "import": true,
	"interface": true, "map": true, "package": true, "range": true, "return": true,
	"select": true, "struct": true, "switch": true, "type": true, "var": true,
}

// ASTFuncDecl holds a C function declaration extracted from Clang's JSON AST,
// with types already converted to Go type strings.
type ASTFuncDecl struct {
	Name       string
	Params     []cParam // reuses cParam: name + goType
	ReturnType string   // Go type string, "" for void
	IsStatic   bool     // true when storageClass == "static"

	// Original C types, kept for generating static-inline wrapper code.
	CReturnType string
	CParams     []cParam // name + C type (goType field holds the C type here)
}

// clangJSONAST writes preamble to a temp file and runs
//
//	clang -Xclang -ast-dump=json -fsyntax-only -target <triple> -I<paths>...
//
// returning the raw JSON bytes.
func clangJSONAST(preamble, triple string, includePaths []string, extraDefines []string) ([]byte, error) {
	tmpDir, err := os.MkdirTemp("", "sigo-cgo-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	preambleFile := filepath.Join(tmpDir, "preamble.c")
	if err := os.WriteFile(preambleFile, []byte(preamble), 0644); err != nil {
		return nil, err
	}

	args := []string{
		"-Xclang", "-ast-dump=json",
		"-fsyntax-only",
	}
	if triple != "" {
		args = append(args, "-target", triple)
	}
	for _, inc := range includePaths {
		args = append(args, "-I"+inc)
	}
	for _, def := range extraDefines {
		args = append(args, "-D"+def)
	}
	args = append(args, preambleFile)

	cmd := exec.Command("clang", args...)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("clang ast-dump failed: %s\n%s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("clang ast-dump failed: %w", err)
	}
	return out, nil
}

// clangASTNode is a minimal representation of a Clang JSON AST node.
type clangASTNode struct {
	Kind         string         `json:"kind"`
	Name         string         `json:"name"`
	MangledName  string         `json:"mangledName"`
	StorageClass string         `json:"storageClass"`
	Inline       bool           `json:"inline"`
	Type         *clangASTType  `json:"type"`
	Inner        []clangASTNode `json:"inner"`
	IsImplicit   bool           `json:"isImplicit"`
}

type clangASTType struct {
	QualType          string `json:"qualType"`
	DesugaredQualType string `json:"desugaredQualType"`
}

// parseClangAST parses the JSON output of clang -ast-dump=json and extracts
// function declarations with their signatures converted to Go types.
// knownTypes maps additional C type names to Go equivalents (e.g. typedef'd structs).
func parseClangAST(jsonData []byte, knownTypes map[string]string) ([]ASTFuncDecl, error) {
	var root clangASTNode
	if err := json.Unmarshal(jsonData, &root); err != nil {
		return nil, fmt.Errorf("parsing clang JSON AST: %w", err)
	}

	// First pass: extract TypedefDecl nodes to resolve typedef'd types
	// (e.g. u8_t → unsigned char, err_t → signed char) that Clang doesn't
	// desugar in FunctionDecl/ParmVarDecl qualType fields.
	for i := range root.Inner {
		node := &root.Inner[i]
		if node.Kind != "TypedefDecl" || node.IsImplicit || node.Type == nil {
			continue
		}
		// Try desugaredQualType first, then qualType.
		var goType string
		var ok bool
		if node.Type.DesugaredQualType != "" {
			goType, ok = convertCTypeForAST(node.Type.DesugaredQualType, knownTypes)
		}
		if !ok {
			goType, ok = convertCTypeForAST(node.Type.QualType, knownTypes)
		}
		if ok {
			// Override _cgo_* placeholders (from typeNames) when the AST
			// resolves the typedef to a concrete Go type (e.g. err_t → int8).
			existing, exists := knownTypes[node.Name]
			if !exists || strings.HasPrefix(existing, "_cgo_") {
				knownTypes[node.Name] = goType
			}
		}
	}

	var funcs []ASTFuncDecl
	for i := range root.Inner {
		node := &root.Inner[i]
		if node.Kind != "FunctionDecl" || node.IsImplicit {
			continue
		}
		if node.Type == nil {
			continue
		}

		// Extract return type from the function's qualType.
		// Format: "rettype (paramtypes...)" e.g. "struct tcp_pcb *(void)"
		cRetType := extractReturnType(node.Type.QualType)
		retGoType, retOk := convertCTypeForAST(cRetType, knownTypes)
		if !retOk && node.Type.DesugaredQualType != "" {
			cRetType = extractReturnType(node.Type.DesugaredQualType)
			retGoType, retOk = convertCTypeForAST(cRetType, knownTypes)
		}
		if !retOk {
			continue
		}

		// Extract parameters from ParmVarDecl children.
		var params []cParam
		var cParams []cParam
		skip := false
		paramIdx := 0
		for j := range node.Inner {
			child := &node.Inner[j]
			if child.Kind != "ParmVarDecl" {
				continue
			}
			if child.Type == nil {
				skip = true
				break
			}

			pName := child.Name
			if pName == "" {
				pName = fmt.Sprintf("arg%d", paramIdx)
			}
			if goKeywords[pName] {
				pName = "_" + pName
			}

			// Use desugaredQualType if available for better type mapping.
			cType := child.Type.QualType
			cTypeForMapping := cType
			if child.Type.DesugaredQualType != "" {
				cTypeForMapping = child.Type.DesugaredQualType
			}

			goType, ok := convertCTypeForAST(cTypeForMapping, knownTypes)
			if !ok {
				// Fall back to qualType.
				goType, ok = convertCTypeForAST(cType, knownTypes)
				if !ok {
					skip = true
					break
				}
			}

			params = append(params, cParam{name: pName, goType: goType})
			cParams = append(cParams, cParam{name: pName, goType: cType})
			paramIdx++
		}
		if skip {
			continue
		}

		funcs = append(funcs, ASTFuncDecl{
			Name:        node.Name,
			Params:      params,
			ReturnType:  retGoType,
			IsStatic:    node.StorageClass == "static",
			CReturnType: cRetType,
			CParams:     cParams,
		})
	}
	return funcs, nil
}

// extractReturnType extracts the return type from a C function qualType string.
// The format is "rettype (paramtypes...)" — we find the outermost '(' from the
// right and take everything before it as the return type.
func extractReturnType(qualType string) string {
	// Find the matching '(' for the final ')'.
	depth := 0
	for i := len(qualType) - 1; i >= 0; i-- {
		switch qualType[i] {
		case ')':
			depth++
		case '(':
			depth--
			if depth == 0 {
				return strings.TrimSpace(qualType[:i])
			}
		}
	}
	return qualType // fallback — shouldn't happen for well-formed qualTypes
}

// convertCTypeForAST normalises a C type string from the AST and returns the
// Go equivalent. Handles const qualifiers, struct/enum/union tags, pointers,
// and function pointer types.
func convertCTypeForAST(cType string, knownTypes map[string]string) (string, bool) {
	t := strings.TrimSpace(cType)

	// Strip const/volatile/restrict qualifiers for mapping purposes.
	t = stripCVR(t)

	// Function pointer types contain "(*)" — always map to unsafe.Pointer.
	if strings.Contains(t, "(*)") {
		return "unsafe.Pointer", true
	}

	// Handle pointer types: if it ends with '*', map to a typed pointer when
	// the base type is known, otherwise use unsafe.Pointer.
	if strings.HasSuffix(t, "*") {
		base := strings.TrimSpace(strings.TrimSuffix(t, "*"))
		base = stripCVR(base)
		for _, prefix := range []string{"struct ", "enum ", "union "} {
			base = strings.TrimPrefix(base, prefix)
		}
		if base != "void" && knownTypes != nil {
			if goName, found := knownTypes[base]; found {
				return "*" + goName, true
			}
		}
		return "unsafe.Pointer", true
	}

	// Strip "struct ", "enum ", "union " prefixes for lookup.
	isEnum := false
	for _, prefix := range []string{"struct ", "enum ", "union "} {
		if strings.HasPrefix(t, prefix) {
			if prefix == "enum " {
				isEnum = true
			}
			t = strings.TrimPrefix(t, prefix)
		}
	}

	goType, ok := convertCTypeWithKnown(t, knownTypes)
	if ok {
		return goType, true
	}

	// Unrecognized enum types default to int32 (C enums are int by default).
	if isEnum {
		return "int32", true
	}

	return "", false
}

// stripCVR removes const, volatile, and restrict qualifiers from a C type string.
func stripCVR(t string) string {
	// Remove leading qualifiers.
	for {
		trimmed := t
		for _, q := range []string{"const ", "volatile ", "restrict "} {
			trimmed = strings.TrimPrefix(trimmed, q)
		}
		if trimmed == t {
			break
		}
		t = trimmed
	}
	// Remove trailing "const" (e.g. "int *const" → "int *").
	t = strings.TrimSuffix(t, " const")
	t = strings.TrimSuffix(t, " volatile")
	t = strings.TrimSuffix(t, " restrict")
	return strings.TrimSpace(t)
}

// GenerateStaticWrappers emits non-static C wrapper functions for each static
// function in funcs whose name appears in referenced. This allows the linker
// to resolve calls from Go to static inline preamble functions.
func GenerateStaticWrappers(funcs map[string]ASTFuncDecl, referenced map[string]bool) string {
	var sb strings.Builder
	sb.WriteString("\n/* sigo: auto-generated wrappers for static functions */\n")
	for name, fn := range funcs {
		if !referenced[name] {
			continue
		}

		// Build parameter list for the wrapper.
		var paramDecls []string
		var argNames []string
		for i, p := range fn.CParams {
			pName := p.name
			if pName == "" {
				pName = fmt.Sprintf("_p%d", i)
			}
			paramDecls = append(paramDecls, fmt.Sprintf("%s %s", p.goType, pName))
			argNames = append(argNames, pName)
		}

		paramStr := strings.Join(paramDecls, ", ")
		if len(paramDecls) == 0 {
			paramStr = "void"
		}
		argsStr := strings.Join(argNames, ", ")

		wrapperName := "__sigo_wrap_" + name
		if fn.CReturnType == "void" || fn.CReturnType == "" {
			fmt.Fprintf(&sb, "%s %s(%s) { %s(%s); }\n",
				"void", wrapperName, paramStr, name, argsStr)
		} else {
			fmt.Fprintf(&sb, "%s %s(%s) { return %s(%s); }\n",
				fn.CReturnType, wrapperName, paramStr, name, argsStr)
		}
	}
	return sb.String()
}
