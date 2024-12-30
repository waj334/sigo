package main

import (
	"encoding/json"
	"omibyte.io/sigo/cmd/csp-gen/generator/importer"
	"os"
)

func main() {
	/*
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
	*/

	d, err := importer.ImportSVD("C:\\Users\\waj33\\Downloads\\stm32f4-svd\\STM32F4_svd\\STM32F4_svd_V2.0\\STM32F405.svd")
	if err != nil {
		panic(err)
	}

	b, err := json.MarshalIndent(&d, "", "  ")
	f, err := os.OpenFile("out.json", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}

	f.Write(b)
	f.Close()
}
