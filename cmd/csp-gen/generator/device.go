package generator

type Device struct {
	Series      string        `json:"series"`
	Flags       AttributeFlag `json:"flags"`
	Variants    []Variant     `json:"variants"`
	Peripherals []Peripheral  `json:"peripherals"`
}

type Variant struct {
	Identifier string      `json:"identifier"`
	Memories   []Memory    `json:"memories"`
	Interrupts []Interrupt `json:"interrupts"`
}

type Memory struct {
	Identifier string        `json:"identifier"`
	Start      uintptr       `json:"start"`
	Size       uintptr       `json:"size"`
	Flags      AttributeFlag `json:"flags"`
}
