package generator

import (
	"fmt"
	"io"
	"strings"
)

type ConstantGroup struct {
	field      *Field
	Identifier string          `json:"identifier"`
	Values     []ConstantValue `json:"values"`
}

type ConstantValue struct {
	ConstantGroup *ConstantGroup
	Identifier    string `json:"identifier"`
	Description   string `json:"description"`
	Value         uint64 `json:"value"`
}

func (c ConstantGroup) TypeName() string {
	r := c.field.register
	f := c.field
	return fmt.Sprintf("Reg%s%s%sType",
		r.Identifier, f.Identifier, c.Identifier)
}

func (c ConstantGroup) WriteTypeDeclaration(output io.Writer) (int, error) {
	return fmt.Fprintf(output, "type %s %s", c.TypeName(), DataType(c.field.Width))
}

func (c ConstantGroup) WriteConstantValues(output io.StringWriter) (int, error) {
	var builder strings.Builder

	fmt.Fprintf(&builder, "const (\n")
	for _, value := range c.Values {
		fmt.Fprintf(&builder, "%s %s = %d\n", value.Identifier, c.TypeName(), value.Value)
	}
	fmt.Fprintf(&builder, ")\n")

	return output.WriteString(builder.String())
}
