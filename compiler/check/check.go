package check

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"regexp"
	"slices"
	"strconv"
)

func CheckAST(fset *token.FileSet, pkg *types.Package, file *ast.File, info *types.Info) error {
	var finalError error
	ast.Inspect(file, func(node ast.Node) bool {
		var checkErr error
		switch node := node.(type) {
		case *ast.CallExpr:
			switch Fun := node.Fun.(type) {
			case *ast.IndexExpr:
				switch X := Fun.X.(type) {
				case *ast.SelectorExpr:
					obj := info.ObjectOf(X.Sel)
					if obj == nil {
						return true
					}

					if obj, ok := obj.(*types.Func); ok {
						switch obj.FullName() {
						case "asm.InlineWithResult":
							checkErr = checkInlineWithResultCall(node, fset, info)
						}
					}
				}
			case *ast.SelectorExpr:
				obj := info.ObjectOf(Fun.Sel)
				if obj == nil {
					return true
				}

				if obj, ok := obj.(*types.Func); ok {
					switch obj.FullName() {
					case "asm.Inline":
						checkErr = checkInlineWithResultCall(node, fset, info)
					}
				}
			}
		}

		if checkErr != nil {
			// Combine with the final error.
			finalError = errors.Join(finalError, checkErr)
		}

		return true
	})

	return finalError
}

func checkInlineWithResultCall(expr *ast.CallExpr, fset *token.FileSet, info *types.Info) error {
	var numInputs, numOutputs int
	var asmStr string
	numConstraints := len(expr.Args) - 1

	// Validate the inline assembly string.
	if lit, ok := expr.Args[0].(*ast.BasicLit); !ok {
		return types.Error{
			Fset: fset,
			Pos:  expr.Args[0].Pos(),
			Msg:  "assembly string must be a string literal",
		}
	} else {
		asmStr = cleanStr(lit.Value)
		if len(asmStr) == 0 {
			return types.Error{
				Fset: fset,
				Pos:  expr.Args[0].Pos(),
				Msg:  "assembly string cannot be empty",
			}
		}
	}

	// Validate constraint arguments.
	if numConstraints > 0 {
		constraintAliases := make([]string, numConstraints)
		for i, argExpr := range expr.Args[1:] {
			switch argExpr := argExpr.(type) {
			case *ast.CallExpr:
				switch Fun := argExpr.Fun.(type) {
				case *ast.SelectorExpr:
					ident, ok := Fun.X.(*ast.Ident)
					if !ok {
						return invalidConstraintArgumentError(fset, argExpr.Pos())
					}

					obj := info.ObjectOf(ident)
					if obj == nil {
						return invalidConstraintArgumentError(fset, argExpr.Pos())
					}

					if pkgObj, ok := obj.(*types.PkgName); !ok || pkgObj.Imported().Path() != "asm" {
						return invalidConstraintArgumentError(fset, argExpr.Pos())
					}

					switch selObj := info.ObjectOf(Fun.Sel).(type) {
					case *types.Func:
						switch selObj.Name() {
						case "In":
							numInputs++
						case "InOut":
							numInputs++
							numOutputs++
						case "Out":
							numOutputs++
						case "Clobber":
							// Do no validation here.
						default:
							return invalidConstraintArgumentError(fset, argExpr.Pos())
						}

						name, err := checkConstraint(argExpr, selObj, fset, info)
						if err != nil {
							return err
						}
						constraintAliases[i] = name
					default:
						return invalidConstraintArgumentError(fset, argExpr.Pos())
					}
				}
			case *ast.SelectorExpr:
				ident, ok := argExpr.X.(*ast.Ident)
				if !ok {
					return invalidConstraintArgumentError(fset, argExpr.Pos())
				}

				obj := info.ObjectOf(ident)
				if obj == nil {
					return invalidConstraintArgumentError(fset, argExpr.Pos())
				}

				// NOTE: Register clobbers are specified via some constant from the `register` package.
				if pkgObj, ok := obj.(*types.PkgName); !ok || pkgObj.Imported().Path() != "asm/register" {
					return invalidConstraintArgumentError(fset, argExpr.Pos())
				}
			default:
				return invalidConstraintArgumentError(fset, argExpr.Pos())
			}
		}

		// Validate usages of aliases in the assembly text.
		re := regexp.MustCompile(`{{([a-zA-Z][a-zA-Z0-9_]*)}}+`)
		matches := re.FindAllStringSubmatch(asmStr, -1)
		for _, match := range matches {
			if !slices.Contains(constraintAliases, match[1]) {
				return types.Error{
					Fset: fset,
					Pos:  expr.Args[0].Pos(),
					Msg:  fmt.Sprintf("undefined alias \"%s\" used in assembly string", match[1]),
					Soft: false,
				}
			}
		}

		// Validate numerical constraint identifiers.
		re = regexp.MustCompile(`\$([0-9]+)`)
		matches = re.FindAllStringSubmatch(asmStr, -1)
		for _, match := range matches {
			index, err := strconv.Atoi(match[1])
			if err != nil {
				return types.Error{
					Fset: fset,
					Pos:  expr.Args[0].Pos(),
					Msg:  fmt.Sprintf("invalid constraint identifier \"%s\"", match[0]),
				}
			}

			if index >= numConstraints {
				return types.Error{
					Fset: fset,
					Pos:  expr.Args[0].Pos(),
					Msg:  fmt.Sprintf("constraint \"%s\" index out of range ", match[0]),
				}
			}
		}
	}

	if Fun, ok := expr.Fun.(*ast.IndexExpr); ok {
		// Validate result type.
		resultType := types.Unalias(info.TypeOf(Fun.Index))

		switch resultType := resultType.(type) {
		case *types.Struct:
			if numOutputs == 1 {
				return types.Error{
					Fset: fset,
					Pos:  Fun.Index.Pos(),
					Msg:  "inline assembly with one output cannot return a struct",
					Soft: false,
				}
			} else if resultType.NumFields() != numOutputs {
				return types.Error{
					Fset: fset,
					Pos:  Fun.Index.Pos(),
					Msg:  "the number of output constraints does not match the number of struct fields",
					Soft: false,
				}
			}
		}
	}

	return nil
}

func checkConstraint(expr *ast.CallExpr, funcObj *types.Func, fset *token.FileSet, info *types.Info) (string, error) {
	var alias string
	var varName string

	funcName := funcObj.Name()

	if funcName == "Clobber" {
		// No additional validation required.
		return "", nil
	}

	// Validate the call args.
	for _, argExpr := range expr.Args {
		switch argExpr := argExpr.(type) {
		case *ast.SelectorExpr:
			switch {
			case exprIs(argExpr, "asm.Register", info):
				return "", types.Error{
					Fset: fset,
					Pos:  argExpr.Pos(),
					Msg:  "a direct register cannot be specified for a constraint",
				}
			case exprIs(argExpr, "asm.clobber", info):
				switch argExpr.Sel.Name {
				case "Reserve":
					if funcName != "Out" && funcName != "InOut" {
						return "", types.Error{
							Fset: fset,
							Pos:  argExpr.Pos(),
							Msg:  "early clobber can only be specified for an output constraint",
						}
					}
				}
			}
		case *ast.CallExpr:
			switch Fun := argExpr.Fun.(type) {
			case *ast.SelectorExpr:
				obj := info.ObjectOf(Fun.Sel)
				if obj == nil {
					return "", invalidConstraintParameterError(fset, argExpr.Pos())
				}

				funcObj, ok := obj.(*types.TypeName)
				if !ok {
					return "", invalidConstraintParameterError(fset, argExpr.Pos())
				}

				modifierFuncName := qualifiedName(funcObj.Name(), funcObj.Pkg())
				switch modifierFuncName {
				case "asm.Alias":
					if len(alias) > 0 {
						return "", types.Error{
							Fset: fset,
							Pos:  argExpr.Pos(),
							Msg:  "constraint already has an alias",
						}
					}

					// The input must be a literal value.
					if lit, ok := argExpr.Args[0].(*ast.BasicLit); !ok {
						return "", invalidConstraintParameterError(fset, argExpr.Pos())
					} else {
						alias = cleanStr(lit.Value)
					}
				case "asm.RegisterClass":
					// This is valid. Do nothing.
				default:
					return "", invalidConstraintParameterError(fset, argExpr.Pos())
				}
			}
		case *ast.UnaryExpr:
			if argExpr.Op == token.AND {
				ident, ok := argExpr.X.(*ast.Ident)
				if !ok {
					return "", invalidConstraintParameterError(fset, argExpr.Pos())
				}
				varName = ident.Name
			}
		case *ast.Ident:
			obj := info.ObjectOf(argExpr)
			if obj == nil {
				return "", invalidConstraintParameterError(fset, argExpr.Pos())
			}

			if len(varName) > 0 {
				return "", types.Error{
					Fset: fset,
					Pos:  argExpr.Pos(),
					Msg:  "only one variable can be specified for any constraint",
				}
			}
			varName = obj.Name()

			if funcName == "Out" {
				return "", types.Error{
					Fset: fset,
					Pos:  argExpr.Pos(),
					Msg:  "output constraint variable must be a pointer",
				}
			}
		default:
			return "", invalidConstraintParameterError(fset, argExpr.Pos())
		}
	}

	// Prefer the alias if one was specified.
	if len(alias) > 0 {
		return alias, nil
	}

	// Otherwise, the constraint will be referred to by the input variable name if one was specified.
	return varName, nil
}

func invalidConstraintArgumentError(fset *token.FileSet, pos token.Pos) error {
	return types.Error{
		Fset: fset,
		Pos:  pos,
		Msg:  "invalid constraint argument",
	}
}

func invalidConstraintParameterError(fset *token.FileSet, pos token.Pos) error {
	return types.Error{
		Fset: fset,
		Pos:  pos,
		Msg:  "invalid parameter to constraint",
	}
}

func exprIs(expr ast.Expr, typeName string, info *types.Info) bool {
	T, ok := info.TypeOf(expr).(*types.Named)
	if !ok || qualifiedName(T.Obj().Name(), T.Obj().Pkg()) != typeName {
		return false
	}
	return true
}
