package peripheral

type Interrupt interface {
	EnableIRQ()
	DisableIRQ()
	SetPriority(priority uint8)
}
