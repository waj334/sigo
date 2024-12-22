package generator

import (
	"fmt"
	"io"
	"strings"
)

type Register struct {
	// Identifier is the name of this register.
	Identifier string

	// Width is the total size of the register in bits.
	Width uintptr `json:"width"`

	// Fields are the individual groupings of bits that represent specific settings.
	Fields []Field `json:"fields"`

	// Description is the text that will go into the API documentation comment.
	Description string `json:"description"`

	// Instances is the list of base addresses for this register type.
	Instances []uintptr `json:"instances"`

	finalized bool
}

func (r *Register) VarName() string {
	return r.Identifier
}

func (r *Register) TypeName() string {
	return fmt.Sprintf("%sType", r.Identifier)
}

func (r *Register) WriteTypeDeclaration(output io.StringWriter) (int, error) {
	var builder strings.Builder

	fmt.Fprintf(&builder, "type %s struct {\n", r.TypeName())

	for _, field := range r.Fields {
		fmt.Fprintf(&builder, "%s %s\n", field.Identifier, field.TypeName())
	}

	fmt.Fprintf(&builder, "}")

	return output.WriteString(builder.String())
}

func (r *Register) WriteConstants(output io.StringWriter) (int, error) {
	var builder strings.Builder
	if len(r.Instances) == 1 {
		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n",
			r.VarName(), r.TypeName(), r.Instances[0])
	} else {
		fmt.Fprintf(&builder, "%s = [%d]*%s{\n", r.VarName(), len(r.Instances), r.TypeName())
		for _, instance := range r.Instances {
			fmt.Fprintf(&builder, "(*%s)(unsafe.Pointer(uintptr(%#x))),\n", r.TypeName(), instance)
		}
		fmt.Fprintf(&builder, "}\n")
	}
	return output.WriteString(builder.String())
}

func (r *Register) WriteMethods(output io.StringWriter) (int, error) {
	var builder strings.Builder
	for _, f := range r.Fields {
		if len(f.Constants.Values) > 0 {
			f.Constants.WriteTypeDeclaration(&builder)
			builder.WriteString("\n\n")

			f.Constants.WriteConstantValues(&builder)
			builder.WriteString("\n")
		}

		f.WriteTypeDeclaration(&builder)
		builder.WriteString("\n")

		f.WriteMethods(&builder)
		builder.WriteString("\n")
	}
	return output.WriteString(builder.String())
}

func (r *Register) String() string {
	return fmt.Sprintf("Register{Identifier: %s, Width: %d, Description: %s}",
		r.Identifier, r.Width, r.Description)
}

func (r *Register) Finalize() {
	if !r.finalized {
		for i := range r.Fields {
			f := &r.Fields[i]
			f.Register = r
			if len(f.Constants.Values) > 0 {
				f.Constants.Field = &r.Fields[i]
				for j := range f.Constants.Values {
					f.Constants.Values[j].ConstantGroup = &f.Constants
				}
			}
		}
		r.finalized = true
	}
}
