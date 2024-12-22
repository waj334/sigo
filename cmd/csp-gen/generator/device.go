package generator

type Device struct {
	Peripherals []Register  `json:"peripherals"`
	Interrupts  []Interrupt `json:"interrupts"`
}

/*type Register struct {
	Fields []Register `json:"registers"`
}*/
