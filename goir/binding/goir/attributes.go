package goir

/*
#include <Go-c/mlir/Dialects.h>
#include <Go-c/mlir/Enums.h>
*/
import "C"
import "pkg.si-go.dev/go-mlir/mlir"

type AsmConstraintAttr struct {
	mlir.Attribute
}

func NewAsmConstraintAttr(
	ctx mlir.Context,
	registerClass string,
	direction AsmConstraintDirection,
	alias string,
	operandIndex int,
	reserve bool,
) AsmConstraintAttr {
	return AsmConstraintAttr{wrapAttribute(C.mlirGoCreateAsmConstraintAttr(
		unwrapContext(ctx),
		unwrapStringRef(mlir.NewStringRef(registerClass)),
		C.MlirGoAsmConstraintDirection(direction),
		unwrapStringRef(mlir.NewStringRef(alias)),
		C.int(operandIndex),
		C.bool(reserve),
	))}
}

func NewCmpFPredicateAttr(ctx mlir.Context, predicate CmpFPredicate) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpFPredicate(unwrapContext(ctx), C.mlirGoCmpFPredicate(predicate)))
}

func NewCmpIPredicateAttr(ctx mlir.Context, predicate CmpIPredicate) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpIPredicate(unwrapContext(ctx), C.mlirGoCmpIPredicate(predicate)))
}

func NewCmpPredicateAttr(ctx mlir.Context, predicate CmpPredicate) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateCmpPredicate(unwrapContext(ctx), C.mlirGoCmpPredicate(predicate)))
}

func NewChanDirectionAttr(ctx mlir.Context, direction ChanDirection) mlir.Attribute {
	return wrapAttribute(C.mlirGoCreateChanDirection(unwrapContext(ctx), C.mlirGoChanDirection(direction)))
}

type ComplexAttr struct {
	mlir.Attribute
}

func NewComplexAttr(ctx mlir.Context, typ mlir.FloatType, real, imag float64) ComplexAttr {
	return ComplexAttr{wrapAttribute(C.mlirGoCreateComplexNumberAttr(
		unwrapContext(ctx),
		unwrapType(typ),
		C.double(real),
		C.double(imag)))}
}

type ScopeAttr struct {
	mlir.Attribute
}

func NewScopeAttr(ctx mlir.Context, parent mlir.AttributeLike, start mlir.LocationLike, end mlir.LocationLike) ScopeAttr {
	return ScopeAttr{
		Attribute: wrapAttribute(C.mlirGoScopeAttrGet(
			unwrapContext(ctx),
			unwrapAttribute(parent),
			unwrapLocation(start),
			unwrapLocation(end),
		)),
	}
}

type DistinctAttr struct {
	mlir.Attribute
}

func NewDistinctAttr(attr mlir.AttributeLike) DistinctAttr {
	return DistinctAttr{
		Attribute: wrapAttribute(C.mlirDistinctAttrGet(
			unwrapAttribute(attr),
		)),
	}
}
