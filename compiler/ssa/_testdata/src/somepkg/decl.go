package somepkg

import "unsafe"

var SomeValue SomeType
var SomePtrValue *SomeType = (*SomeType)(unsafe.Pointer(uintptr(0)))

const SomeConstAliasTypeValue SomeTypeAlias = 0

type SomeType struct {
	Member    SomeMemberType
	MemberPtr *SomeMemberType
	A         int
}

type SomeTypeAlias int

func (s SomeType) Method() {

}

func (s *SomeType) PtrMethod() {

}

type SomeMemberType struct{}

func (s SomeMemberType) Method() {

}

func (s *SomeMemberType) PtrMethod() {

}

func (s SomeTypeAlias) Method()     {}
func (s *SomeTypeAlias) PtrMethod() {}

func SomeFunc() {}
