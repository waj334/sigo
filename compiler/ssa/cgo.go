package ssa

import (
	"fmt"
	"regexp"
	"strings"
)

// cTypeToGo maps C primitive type names (normalised, no trailing whitespace) to
// their Go equivalents. Pointer types are handled separately in convertCType.
var cTypeToGo = map[string]string{
	"void":         "",
	"int":          "int32",
	"signed int":   "int32",
	"unsigned int": "uint32",
	"unsigned":     "uint32",

	"long":               "int32",
	"signed long":        "int32",
	"unsigned long":      "uint32",
	"long long":          "int64",
	"signed long long":   "int64",
	"unsigned long long": "uint64",

	"short":          "int16",
	"signed short":   "int16",
	"unsigned short": "uint16",
	"char":           "int8",
	"signed char":    "int8",
	"unsigned char":  "uint8",
	"float":          "float32",
	"double":         "float64",
	"int8_t":         "int8",
	"int16_t":        "int16",
	"int32_t":        "int32",
	"int64_t":        "int64",
	"uint8_t":        "uint8",
	"uint16_t":       "uint16",
	"uint32_t":       "uint32",
	"uint64_t":       "uint64",
	"size_t":         "uintptr",
	"uintptr_t":      "uintptr",
	"ptrdiff_t":      "int",
	"bool":           "bool",
	"_Bool":          "bool",
}

// cTypeAliases is the ordered list of C primitive types for which we emit
// Go type aliases in the generated file (as _cgo_<cName> = <goType>).
// Only include types a user might write as C.<type>.
var cTypeAliases = []struct{ cName, goType string }{
	{"void", ""},
	{"int", "int32"},
	{"uint", "uint32"},
	{"long", "int32"},
	{"short", "int16"},
	{"char", "int8"},
	{"uchar", "uint8"},
	{"float", "float32"},
	{"double", "float64"},
	{"schar", "int8"},
	{"int8_t", "int8"},
	{"int16_t", "int16"},
	{"int32_t", "int32"},
	{"int64_t", "int64"},
	{"uint8_t", "uint8"},
	{"uint16_t", "uint16"},
	{"uint32_t", "uint32"},
	{"uint64_t", "uint64"},
	{"size_t", "uintptr"},
	{"uintptr_t", "uintptr"},
	{"ptrdiff_t", "int"},
	{"bool", "bool"},
}

type cParam struct {
	name   string
	goType string
}

type cFuncDecl struct {
	name       string
	params     []cParam
	returnType string
}

type cStructDecl struct {
	name   string
	fields []cParam // reuses cParam: name + goType
}

// cFuncRe matches simple C function declarations and definitions.
// Group 1: return type (possibly multi-word, may include *)
// Group 2: function name
// Group 3: parameter list (everything inside the outer parens)
var cFuncRe = regexp.MustCompile(`(?m)^[ \t]*([\w][\w\s*]*?)\s+(\w+)\s*\(([^)]*)\)\s*(?:;|\{)`)

// cStructRe matches typedef struct declarations.
// Group 1: struct body (fields between braces)
// Group 2: typedef name
var cStructRe = regexp.MustCompile(`(?s)typedef\s+struct\s*(?:\w+\s*)?\{([^}]*)\}\s*(\w+)\s*;`)

// convertCType normalises a C type string and returns the corresponding Go type.
// Returns "" for void, "unsafe.Pointer" for pointer types, and an empty string
// plus ok==false when the type cannot be mapped.
func convertCType(cType string) (goType string, ok bool) {
	return convertCTypeWithKnown(cType, nil)
}

// convertCTypeWithKnown is like convertCType but also checks knownTypes for
// user-defined type names (e.g. typedef'd structs).
func convertCTypeWithKnown(cType string, knownTypes map[string]string) (goType string, ok bool) {
	t := strings.TrimSpace(cType)
	// Collapse internal whitespace runs to a single space.
	t = strings.Join(strings.Fields(t), " ")

	// Handle pointer types: anything ending in * (after normalisation).
	if strings.HasSuffix(t, "*") {
		return "unsafe.Pointer", true
	}

	// Direct lookup in primitive types.
	if g, found := cTypeToGo[t]; found {
		return g, true
	}

	// Check user-defined types (typedef'd structs, etc.).
	if knownTypes != nil {
		if g, found := knownTypes[t]; found {
			return g, true
		}
	}

	return "", false
}

// parseCStructs extracts typedef struct declarations from a C preamble.
// It returns one cStructDecl per matched typedef, skipping any whose field
// types cannot be mapped to Go.
func parseCStructs(preamble string) []cStructDecl {
	matches := cStructRe.FindAllStringSubmatch(preamble, -1)
	var structs []cStructDecl
	for _, m := range matches {
		body := strings.TrimSpace(m[1])
		name := strings.TrimSpace(m[2])

		var fields []cParam
		skip := false
		for _, line := range strings.Split(body, ";") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			tokens := strings.Fields(line)
			if len(tokens) < 2 {
				continue
			}
			fieldName := tokens[len(tokens)-1]
			fieldType := strings.Join(tokens[:len(tokens)-1], " ")
			goType, mapped := convertCType(fieldType)
			if !mapped {
				skip = true
				break
			}
			fields = append(fields, cParam{name: fieldName, goType: goType})
		}
		if skip || len(fields) == 0 {
			continue
		}
		structs = append(structs, cStructDecl{name: name, fields: fields})
	}
	return structs
}

// parseCFunctions extracts simple C function declarations from a C preamble.
// It returns one cFuncDecl per matched declaration, skipping any whose
// parameter or return types cannot be mapped to Go.
// knownTypes maps additional C type names to their Go equivalents (e.g.
// typedef'd structs: "testStruct" → "_cgo_testStruct").
func parseCFunctions(preamble string, knownTypes map[string]string) []cFuncDecl {
	matches := cFuncRe.FindAllStringSubmatch(preamble, -1)
	var funcs []cFuncDecl
	for _, m := range matches {
		retC := strings.TrimSpace(m[1])
		name := strings.TrimSpace(m[2])
		paramStr := strings.TrimSpace(m[3])

		retGo, ok := convertCTypeWithKnown(retC, knownTypes)
		if !ok {
			continue
		}

		var params []cParam
		skip := false
		if paramStr != "" && paramStr != "void" {
			for _, p := range strings.Split(paramStr, ",") {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				// Split off the last token as the parameter name (if present).
				tokens := strings.Fields(p)
				var pName, pType string
				if len(tokens) == 1 {
					// No name, only type.
					pType = tokens[0]
					pName = fmt.Sprintf("arg%d", len(params))
				} else {
					// Last token is the name, the rest is the type.
					// Handle pointer: name might start with *.
					if strings.HasPrefix(tokens[len(tokens)-1], "*") {
						// e.g. "int *ptr" — name is "ptr", type is "int *"
						pName = strings.TrimLeft(tokens[len(tokens)-1], "*")
						pType = strings.Join(tokens[:len(tokens)-1], " ") + " *"
					} else {
						pName = tokens[len(tokens)-1]
						pType = strings.Join(tokens[:len(tokens)-1], " ")
					}
				}
				goType, mapped := convertCTypeWithKnown(pType, knownTypes)
				if !mapped {
					skip = true
					break
				}
				params = append(params, cParam{name: pName, goType: goType})
			}
		}
		if skip {
			continue
		}
		funcs = append(funcs, cFuncDecl{name: name, params: params, returnType: retGo})
	}
	return funcs
}

// generateCGoFile produces the content of a synthetic Go source file that:
//   - declares type aliases for common C primitive types (_cgo_int = int32, etc.)
//   - declares Go struct types for C typedef structs referenced by the user
//   - declares body-less function stubs with //sigo:extern pragmas for each C
//     function in funcs whose name appears in the referenced cgoNames set.
//
// pkgName is the Go package name (e.g. "main").
func generateCGoFile(pkgName string, funcs []cFuncDecl, structs []cStructDecl, cgoNames []string) string {
	// Build a set of referenced CGo names for fast lookup.
	referenced := make(map[string]bool, len(cgoNames))
	for _, n := range cgoNames {
		referenced[n] = true
	}

	var sb strings.Builder
	sb.WriteString("package ")
	sb.WriteString(pkgName)
	sb.WriteString("\n\nimport \"unsafe\"\n\n")

	// Emit type aliases for C primitive types.
	sb.WriteString("type (\n")
	for _, alias := range cTypeAliases {
		if alias.goType == "" {
			continue // skip void — it has no Go representation
		}
		fmt.Fprintf(&sb, "\t_cgo_%s = %s\n", alias.cName, alias.goType)
	}
	sb.WriteString(")\n\n")

	// Suppress "imported and not used" for unsafe.
	sb.WriteString("var _ = unsafe.Pointer(nil)\n\n")

	// Emit struct type definitions for referenced C typedef structs.
	for _, s := range structs {
		if !referenced[s.name] {
			continue
		}
		fmt.Fprintf(&sb, "type _cgo_%s struct {\n", s.name)
		for _, f := range s.fields {
			fmt.Fprintf(&sb, "\t%s %s\n", f.name, f.goType)
		}
		sb.WriteString("}\n\n")
	}

	// Emit function stubs for referenced C functions whose signatures were parsed.
	for _, fn := range funcs {
		if !referenced[fn.name] {
			continue
		}
		// //sigo:extern <goName> <cLinkName>
		fmt.Fprintf(&sb, "//sigo:extern _cgo_%s %s\n", fn.name, fn.name)
		sb.WriteString("func _cgo_")
		sb.WriteString(fn.name)
		sb.WriteString("(")
		for i, p := range fn.params {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(p.name)
			sb.WriteString(" ")
			sb.WriteString(p.goType)
		}
		sb.WriteString(")")
		if fn.returnType != "" {
			sb.WriteString(" ")
			sb.WriteString(fn.returnType)
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// generateCGoFileFromAST produces the content of a synthetic Go source file
// from AST-extracted function declarations. It emits type aliases for common
// C types, struct definitions, opaque struct placeholders, and function stubs
// for each C function whose name appears in the cgoNames set.
// staticFuncs maps function names to their AST declarations for static
// functions that need __sigo_wrap_ link names.
// typeNames identifies cgoNames that are used in type positions (type aliases,
// field types) rather than as callable functions or variables.
func generateCGoFileFromAST(pkgName string, funcs []ASTFuncDecl, structs []cStructDecl, cgoNames []string, typeNames map[string]bool, staticFuncs map[string]bool, knownTypes map[string]string) string {
	referenced := make(map[string]bool, len(cgoNames))
	for _, n := range cgoNames {
		referenced[n] = true
	}

	var sb strings.Builder
	sb.WriteString("package ")
	sb.WriteString(pkgName)
	sb.WriteString("\n\nimport \"unsafe\"\n\n")

	// Emit type aliases for C primitive types.
	sb.WriteString("type (\n")
	for _, alias := range cTypeAliases {
		if alias.goType == "" {
			continue
		}
		fmt.Fprintf(&sb, "\t_cgo_%s = %s\n", alias.cName, alias.goType)
	}
	sb.WriteString(")\n\n")

	// Suppress "imported and not used" for unsafe.
	sb.WriteString("var _ = unsafe.Pointer(nil)\n\n")

	// Emit struct type definitions for referenced C typedef structs (defined in preamble text).
	resolvedStructs := make(map[string]bool)
	for _, s := range structs {
		if !referenced[s.name] {
			continue
		}
		fmt.Fprintf(&sb, "type _cgo_%s struct {\n", s.name)
		for _, f := range s.fields {
			fmt.Fprintf(&sb, "\t%s %s\n", f.name, f.goType)
		}
		sb.WriteString("}\n\n")
		resolvedStructs[s.name] = true
	}

	// Emit opaque struct placeholders for type names from included headers
	// that are not primitive aliases and not already emitted above.
	primitiveSet := make(map[string]bool)
	for _, alias := range cTypeAliases {
		primitiveSet[alias.cName] = true
	}
	for _, name := range cgoNames {
		if typeNames[name] && !primitiveSet[name] && !resolvedStructs[name] {
			// If the typedef was resolved to a primitive Go type, emit a
			// type alias (e.g. type _cgo_err_t = int8) instead of an
			// opaque struct placeholder.
			if goType, ok := knownTypes[name]; ok && !strings.HasPrefix(goType, "_cgo_") && !strings.HasPrefix(goType, "*_cgo_") {
				fmt.Fprintf(&sb, "type _cgo_%s = %s\n\n", name, goType)
			} else {
				fmt.Fprintf(&sb, "type _cgo_%s struct{}\n\n", name)
			}
		}
	}

	// Emit function stubs for referenced C functions.
	for _, fn := range funcs {
		if !referenced[fn.Name] {
			continue
		}

		// For static functions, use the __sigo_wrap_ link name so the
		// linker finds the generated non-static wrapper.
		linkName := fn.Name
		if staticFuncs[fn.Name] {
			linkName = "__sigo_wrap_" + fn.Name
		}

		// //sigo:extern _cgo_<name> <linkName>
		fmt.Fprintf(&sb, "//sigo:extern _cgo_%s %s\n", fn.Name, linkName)
		sb.WriteString("func _cgo_")
		sb.WriteString(fn.Name)
		sb.WriteString("(")
		for i, p := range fn.Params {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(p.name)
			sb.WriteString(" ")
			sb.WriteString(p.goType)
		}
		sb.WriteString(")")
		if fn.ReturnType != "" {
			sb.WriteString(" ")
			sb.WriteString(fn.ReturnType)
		}
		sb.WriteString("\n")
	}

	return sb.String()
}
