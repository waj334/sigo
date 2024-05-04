// RUN: FileCheck %s

package main

type interfaceA interface {
	a()
	b()
	c()
}

type interfaceB interface {
	a()
	b()
	c()
	d()
}

type interfaceC interface {
	interfaceA
	d()
}

type structA struct{}

func (s *structA) a() {
}

func (s *structA) b() {
}

func (s *structA) c() {
}

type structAA struct{}

func (s structAA) a() {
}

func (s structAA) b() {
}

func (s structAA) c() {
}

func interfaceAssignment(a interfaceA, b interfaceB, c interfaceC) {
	a = b
	a = c
	b = c
	c = b
}

func interfaceReturn(b interfaceB) (interfaceA, interfaceC) {
	return b, b
}

func interfaceAssignmentAs(a structA, aa structAA, i interfaceA) {
	//i = &a
	i = aa
}

func interfaceReturnAs(a structA, aa structAA) (interfaceA, interfaceA) {
	return &a, aa
}

func nilError() error {
	return nil
}

func assignNilError(err error) {
	err = nil
}

func assignDeclareNilError() error {
	err := error(nil)
	return err
}
