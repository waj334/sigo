package compiler

import (
	"context"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ssa"
)

type Instruction interface {
	Generate(ctx context.Context) Value
	Pos() token.Pos
	String() string
}

type instructionCommon struct {
	cc  *Compiler
	pos token.Pos
	str string
}

func (i instructionCommon) Pos() token.Pos {
	return i.pos
}

func (i instructionCommon) String() string {
	return i.str
}

type Expression interface {
	Instruction
	Type() types.Type
}

type expressionBase struct {
	instructionCommon
	goType types.Type
}

func (e expressionBase) Type() types.Type {
	return e.goType
}

func (c *Compiler) createExpression2(value ssa.Value) (expr Expression) {
	switch t := value.(type) {
	case *ssa.Alloc:
		expr = &Alloc{
			expressionBase: expressionBase{
				instructionCommon: instructionCommon{
					cc:  c,
					pos: value.Pos(),
					str: value.String(),
				},
				goType: value.Type(),
			},
			Heap:    t.Heap,
			Comment: t.Comment,
		}
	}

	return expr
}
