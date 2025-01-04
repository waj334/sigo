package device

import (
	"fmt"
)

type Field struct {
	register *Register

	// Identifier is the identifier of the register.
	Identifier string `json:"identifier"`

	// Width is the size of the field in bits.
	Width uintptr `json:"width"`

	// Offset is position within the register in bits.
	Offset uintptr `json:"offset"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlags `json:"flags"`

	// Description is the text that will go into the API documentation comment.
	Description string `json:"description"`

	// Constants are the predefined values that this field accepts.
	Constants *ConstantGroup `json:"constants,omitempty"`
}

func (f *Field) Register() Register {
	return *f.register
}

func (f *Field) String() string {
	return fmt.Sprintf("Field{Register %s, Identifier: %s, Width: %d, Offset: %d, Flags: %s, Description: %s}",
		f.register.Identifier, f.Identifier, f.Width, f.Offset, f.Flags, f.Description)
}
