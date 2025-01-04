package device

type Device struct {
	Series      string         `json:"series"`
	BuildTags   []string       `json:"buildTags,omitempty"`
	Flags       AttributeFlags `json:"flags"`
	Variants    []Variant      `json:"variants"`
	Peripherals []Peripheral   `json:"peripherals"`

	finalized bool
}

func (d *Device) Finalize() {
	if !d.finalized {
		d.finalized = true
		for i := range d.Peripherals {
			d.Peripherals[i].Finalize()
		}
	}
}

type Variant struct {
	Identifier string      `json:"identifier"`
	Memories   []Memory    `json:"memories"`
	Interrupts []Interrupt `json:"interrupts"`
}

type MemoryType string

const (
	MemoryUnknown MemoryType = "unknown"
	MemoryRAM                = "RAM"
	MemoryFlash              = "FLASH"
	MemoryIO                 = "IO"
	MemoryFuses              = "FUSES"
	MemoryUser               = "USER"
	MemoryOther              = "OTHER"
)

type Memory struct {
	Identifier string         `json:"identifier"`
	Type       MemoryType     `json:"type"`
	Start      Address        `json:"start"`
	Size       Address        `json:"size"`
	Flags      AttributeFlags `json:"flags"`
}
