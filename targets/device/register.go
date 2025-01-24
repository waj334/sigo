package device

import (
	"fmt"
)

type Positionable interface {
	OffsetInBytes() uintptr
	TotalWidthInBits() uintptr
	TotalWidthInBytes() uintptr
	SizeInBytes() uintptr
}

type RegisterGroup struct {
	Identifier  string          `json:"identifier"`
	Reference   string          `json:"reference,omitempty"`
	Description string          `json:"description,omitempty"`
	Offset      Address         `json:"offset"`
	Size        Address         `json:"size"`
	Count       int             `json:"count"`
	Groups      []RegisterGroup `json:"groups,omitempty"`
	Registers   []Register      `json:"registers,omitempty"`
	Merge       *bool           `json:"merge,omitempty"`

	// Embed the type into the peripheral struct.
	Embed *bool `json:"embed"`

	peripheral *Peripheral
	finalized  bool
}

func (r *RegisterGroup) Peripheral() *Peripheral {
	return r.peripheral
}

func (r *RegisterGroup) OffsetInBytes() uintptr {
	return uintptr(r.Offset)
}

func (r *RegisterGroup) TotalWidthInBits() uintptr {
	return r.TotalWidthInBytes() * 8
}

func (r *RegisterGroup) TotalWidthInBytes() uintptr {
	return uintptr(r.Size * Address(max(1, r.Count)))
}

func (r *RegisterGroup) SizeInBytes() uintptr {
	return uintptr(r.Size)
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

func (r *RegisterGroup) GetMerge() bool {
	if r.Merge != nil {
		return *r.Merge
	}
	return false
}

type Register struct {
	// Identifier is the name of this register.
	Identifier string `json:"identifier"`

	// Width is the total size of the register in bits.
	Width uintptr `json:"width"`

	Offset Address `json:"offset"`

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

func (r *Register) OffsetInBytes() uintptr {
	return uintptr(r.Offset)
}

func (r *Register) TotalWidthInBits() uintptr {
	return r.Width * max(1, uintptr(r.Count))
}

func (r *Register) TotalWidthInBytes() uintptr {
	return (r.Width * max(1, uintptr(r.Count))) / 8
}

func (r *Register) SizeInBytes() uintptr {
	return uintptr(r.Width) / 8
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
