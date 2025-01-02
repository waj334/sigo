package device

type ConstantGroup struct {
	field      *Field
	Identifier string          `json:"identifier"`
	Values     []ConstantValue `json:"values"`
}

func (c ConstantGroup) Field() Field {
	return *c.field
}

type ConstantValue struct {
	constantGroup *ConstantGroup
	Identifier    string `json:"identifier"`
	Description   string `json:"description"`
	Value         uint64 `json:"value"`
}

func (c ConstantValue) ConstantGroup() ConstantGroup {
	return *c.constantGroup
}
