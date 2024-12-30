package device

type Interrupt struct {
	Identifier  string `json:"identifier"`
	Description string `json:"description"`
	Number      int    `json:"number"`
}
