package device

type Peripheral struct {
	Identifier  string   `json:"identifier"`
	Group       string   `json:"group,omitempty"`
	BaseAddress *Address `json:"baseAddress,omitempty"`
	Description string   `json:"description"`

	// Flags describes the characteristics of the register (r, w, rw, etc...)
	Flags AttributeFlags `json:"flags"`

	// Instances is the list of base addresses for this register type.
	Instances []Address `json:"instances,omitempty"`

	RegisterGroups []RegisterGroup `json:"registerGroups,omitempty"`
	Interrupts     []Interrupt     `json:"interrupts,omitempty"`

	BuildTags []string `json:"buildTags,omitempty"`

	finalized bool
}

func (p *Peripheral) Finalize() {
	if !p.finalized {
		p.finalized = true
		for i := range p.RegisterGroups {
			p.RegisterGroups[i].peripheral = p
			p.RegisterGroups[i].Finalize()
		}
	}
}
