package somepkg

var SomeValue SomeType
var SomePtrValue *SomeType

type SomeType struct {
	Member SomeMemberType
	A      int
}

func (s SomeType) Method() {

}

func (s *SomeType) PtrMethod() {

}

type SomeMemberType struct{}

func (s SomeMemberType) Method() {

}

func (s *SomeMemberType) PtrMethod() {

}

func SomeFunc() {}
