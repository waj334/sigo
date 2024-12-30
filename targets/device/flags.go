package device

import "slices"

/*
type AttributeFlag uint8

const (
	NotSet    AttributeFlag = 0
	Read      AttributeFlag = 0b0000_0001
	Write     AttributeFlag = 0b0000_0010
	ReadWrite AttributeFlag = Read | Write
	MainFlash AttributeFlag = 0b0000_0100
	MainRAM   AttributeFlag = 0b0000_1000
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
*/

type AttributeFlag string

const (
	Read         AttributeFlag = "R"
	Write                      = "W"
	PrimaryFlash               = "FLASH"
	PrimaryRam                 = "RAM"
)

type AttributeFlags []AttributeFlag

func (a *AttributeFlags) Set(flags ...AttributeFlag) {
	for _, flag := range flags {
		if !slices.Contains(*a, flag) {
			*a = append(AttributeFlags{flag}, *a...)
		}
	}
}

func (a *AttributeFlags) Unset(flag AttributeFlag) {
	slices.DeleteFunc(*a, func(v AttributeFlag) bool {
		return v == flag
	})
}

func (a *AttributeFlags) Clear() {
	*a = AttributeFlags{}
}

func (a *AttributeFlags) IsSet(flags ...AttributeFlag) bool {
	for _, flag := range flags {
		if !slices.Contains(*a, flag) {
			return false
		}
	}
	return true
}

func (a *AttributeFlags) IsUnset() bool {
	return len(*a) == 0
}
