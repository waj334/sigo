package generator

import (
	"fmt"
	"io"
	"strings"
)

type Peripheral struct {
	Identifier  string  `json:"identifier"`
	BaseAddress uintptr `json:"baseAddress"`
	Description string  `json:"description"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlag `json:"flags"`

	// Instances is the list of base addresses for this register type.
	Instances []uintptr `json:"instances,omitempty"`

	Registers  []Register  `json:"registers"`
	Interrupts []Interrupt `json:"interrupts,omitempty"`
}

func (p *Peripheral) VarName() string {
	return p.Identifier
}

func (p *Peripheral) TypeName() string {
	return fmt.Sprintf("Peripheral%sType", p.Identifier)
}

func (p *Peripheral) WriteConstants(output io.StringWriter) (int, error) {
	var builder strings.Builder
	if len(p.Instances) == 0 {
		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n",
			p.VarName(), p.TypeName(), p.BaseAddress)
	} else if len(p.Instances) == 1 {
		fmt.Fprintf(&builder, "%s = (*%s)(unsafe.Pointer(uintptr(%#x)))\n",
			p.VarName(), p.TypeName(), p.Instances[0])
	} else {
		fmt.Fprintf(&builder, "%s = [%d]*%s{\n", p.VarName(), len(p.Instances), p.TypeName())
		for _, instance := range p.Instances {
			fmt.Fprintf(&builder, "(*%s)(unsafe.Pointer(uintptr(%#x))),\n", p.TypeName(), instance)
		}
		fmt.Fprintf(&builder, "}\n")
	}
	return output.WriteString(builder.String())
}
