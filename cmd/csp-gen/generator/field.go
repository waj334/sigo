package generator

import (
	"fmt"
	"github.com/sirkon/go-format/v2"
	"io"
	"strings"
)

type Field struct {
	register *Register

	// Identifier is the identifier of the register.
	Identifier string `json:"name"`

	// Width is the size of the field in bits.
	Width uintptr `json:"width"`

	// Offset is position within the register in bits.
	Offset uintptr `json:"offset"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlag `json:"flags"`

	// Description is the text that will go into the API documentation comment.
	Description string `json:"description"`

	// Constants are the predefined values that this field accepts.
	Constants *ConstantGroup `json:"constants,omitempty"`
}

func (f *Field) TypeName() string {
	return fmt.Sprintf("%sType", f.Identifier)
}

func (f *Field) WriteTypeDeclaration(output io.StringWriter) (int, error) {
	var builder strings.Builder
	fmt.Fprintf(&builder, "type %s uint%d\n", f.TypeName(), NextPow2(f.Width))
	return output.WriteString(builder.String())
}

func (f *Field) WriteMethods(output io.StringWriter) (int, error) {
	var builder strings.Builder
	params := fieldParams(*f)
	if f.Flags.IsSet(Read) {
		if f.Width == 1 {
			// Output an API that uses the `bool` data type  as the return type.
			fmt.Fprintf(&builder, "%s\n", format.Formatm(boolGetter, params))
		} else {
			// Output an API that uses an integer data type of the necessary bit width as the return type.
			fmt.Fprintf(&builder, "%s\n", format.Formatm(intGetter, params))
		}
		builder.WriteString("\n")
	}

	if f.Flags.IsSet(Write) {
		if f.Width == 1 {
			// Output an API that uses the `bool` data type  as the return type.
			fmt.Fprintf(&builder, "%s\n", format.Formatm(boolSetter, params))
		} else {
			// Output an API that uses an integer data type of the necessary bit width as the return type.
			fmt.Fprintf(&builder, "%s\n", format.Formatm(intSetter, params))
		}
		builder.WriteString("\n")
	}

	return output.WriteString(builder.String())
}

func (f *Field) String() string {
	return fmt.Sprintf("Field{Register %s, Identifier: %s, Width: %d, Offset: %d, Flags: %s, Description: %s}",
		f.register.Identifier, f.Identifier, f.Width, f.Offset, f.Flags, f.Description)
}
