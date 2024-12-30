package generator

type AttributeFlag uint8

const (
	NotSet    AttributeFlag = 0
	Read      AttributeFlag = 0b0000_0001
	Write     AttributeFlag = 0b0000_0010
	ReadWrite AttributeFlag = Read | Write
)

func (f *AttributeFlag) Set(flag AttributeFlag) {
	*f |= flag
}

func (f *AttributeFlag) Unset(flag AttributeFlag) {
	*f &^= flag
}

func (f *AttributeFlag) Clear() {
	*f = 0
}
func (f *AttributeFlag) IsSet(flag AttributeFlag) bool {
	return *f&flag != 0
}
