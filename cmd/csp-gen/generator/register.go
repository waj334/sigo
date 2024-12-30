package generator

import (
	"fmt"
	"io"
	"strings"
)

type Register struct {
	// Identifier is the name of this register.
	Identifier string `json:"identifier"`

	// Width is the total size of the register in bits.
	Width uintptr `json:"width"`

	// Fields are the individual groupings of bits that represent specific settings.
	Fields []Field `json:"fields"`

	// Description is the text that will go into the API documentation comment.
	Description string `json:"description"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlag `json:"flags"`

	finalized bool
}

func (r *Register) TypeName() string {
	return fmt.Sprintf("Reg%sType", r.Identifier)
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
			f.register = r
			if len(f.Constants.Values) > 0 {
				f.Constants.field = &r.Fields[i]
				for j := range f.Constants.Values {
					f.Constants.Values[j].ConstantGroup = f.Constants
				}
			}
		}
		r.finalized = true
	}
}
