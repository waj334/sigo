package device

import (
	"fmt"
)

type RegisterGroup struct {
	Identifier  string          `json:"identifier"`
	Reference   string          `json:"reference,omitempty"`
	Description string          `json:"description,omitempty"`
	Count       int             `json:"count"`
	Groups      []RegisterGroup `json:"groups,omitempty"`
	Registers   []Register      `json:"registers,omitempty"`

	peripheral *Peripheral
	finalized  bool
}

func (r *RegisterGroup) Finalize() {
	if !r.finalized {
		r.finalized = true
		for i := range r.Groups {
			r.Groups[i].peripheral = r.peripheral
			r.Groups[i].Finalize()
		}

		for i := range r.Registers {
			r.Registers[i].peripheral = r.peripheral
			r.Registers[i].Finalize()
		}
	}
}

type Register struct {
	// Identifier is the name of this register.
	Identifier string `json:"identifier"`

	// Width is the total size of the register in bits.
	Width uintptr `json:"width"`

	Offset uintptr `json:"offset"`

	Count int `json:"count"`

	// Fields are the individual groupings of bits that represent specific settings.
	Fields []Field `json:"fields"`

	// Description is the text that will go into the API documentation comment.
	Description string `json:"description"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlags `json:"flags"`

	peripheral *Peripheral
	finalized  bool
}

func (r *Register) Peripheral() Peripheral {
	return *r.peripheral
}

func (r *Register) TypeName() string {
	return fmt.Sprintf("Reg%s%sType", r.peripheral.Identifier, r.Identifier)
}

func (r *Register) TotalWidth() uintptr {
	return r.Width * max(1, uintptr(r.Count))
}

func (r *Register) String() string {
	return fmt.Sprintf("Register{Identifier: %s, Offset: %d, Width: %d, Description: %s}",
		r.Identifier, r.Offset, r.Width, r.Description)
}

func (r *Register) Finalize() {
	if !r.finalized {
		for i := range r.Fields {
			f := &r.Fields[i]
			f.register = r
			if f.Constants != nil {
				f.Constants.field = &r.Fields[i]
				for j := range f.Constants.Values {
					f.Constants.Values[j].constantGroup = f.Constants
				}
			}
		}
		r.finalized = true
	}
}
