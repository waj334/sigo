package main

import (
	"go/format"
	"omibyte.io/sigo/cmd/csp-gen/generator"
	"strings"
)

func main() {
	var builder strings.Builder

	r := generator.Register{
		Identifier:  "OSCCTRL",
		Width:       32,
		Description: "Test OSC register",
		Fields: []generator.Field{
			{
				Identifier:  "INTENCLR",
				Width:       32,
				Offset:      0,
				Flags:       generator.ReadWrite,
				Description: "Clear interrupt enable flag",
				Constants: generator.ConstantGroup{
					Identifier: "TestConst",
					Values: []generator.ConstantValue{
						{
							Identifier: "Value0",
							Value:      0,
						},
						{
							Identifier: "Value1",
							Value:      1,
						},
						{
							Identifier: "Value2",
							Value:      2,
						},
					},
				},
			},
		},
		Instances: []uintptr{
			0x40001000,
		},
	}
	r.Finalize()

	r.WriteTypeDeclaration(&builder)
	builder.WriteString("\n\n")

	builder.WriteString("var (\n")
	r.WriteConstants(&builder)
	builder.WriteString(")\n")
	builder.WriteString("\n")

	r.WriteMethods(&builder)
	builder.WriteString("\n\n")

	src, err := format.Source([]byte(builder.String()))
	if err != nil {
		panic(err)
	}
	println(string(src))
}
