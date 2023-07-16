package reflect

import "unsafe"

func Append(s Value, x ...Value) Value {
	panic("not implemented")
}

func AppendSlice(s, t Value) Value {
	panic("not implemented")
}

func Indirect(v Value) Value {
	panic("not implemented")
}

func MakeChan(typ Type, buffer int) Value {
	panic("not implemented")
}

func MakeFunc(typ Type, fn func(args []Value) (results []Value)) Value {
	panic("not implemented")
}

func MakeMap(typ Type) Value {
	panic("not implemented")
}

func MakeMapWithSize(typ Type, n int) Value {
	panic("not implemented")
}

func MakeSlice(typ Type, len, cap int) Value {
	panic("not implemented")
}

func New(typ Type) Value {
	panic("not implemented")
}

func NewAt(typ Type, p unsafe.Pointer) Value {
	panic("not implemented")
}

func Select(cases []SelectCase) (chosen int, recv Value, recvOK bool) {
	panic("not implemented")
}

func ValueOf(i any) Value {
	panic("not implemented")
}

func Zero(typ Type) Value {
	panic("not implemented")
}

type MapIter struct {
}

func (iter *MapIter) Key() Value {
	panic("not implemented")
}

func (iter *MapIter) Next() bool {
	panic("not implemented")
}

func (iter *MapIter) Reset(v Value) {
	panic("not implemented")
}

func (iter *MapIter) Value() Value {
	panic("not implemented")
}

type Value struct {
}

func (v Value) Addr() Value {
	panic("not implemented")
}

func (v Value) Bool() bool {
	panic("not implemented")
}

func (v Value) Bytes() []byte {
	panic("not implemented")
}

func (v Value) Call(in []Value) []Value {
	panic("not implemented")
}

func (v Value) CallSlice(in []Value) []Value {
	panic("not implemented")
}

func (v Value) CanAddr() bool {
	panic("not implemented")
}

func (v Value) CanComplex() bool {
	panic("not implemented")
}

func (v Value) CanConvert(t Type) bool {
	panic("not implemented")
}

func (v Value) CanFloat() bool {
	panic("not implemented")
}

func (v Value) CanInt() bool {
	panic("not implemented")
}

func (v Value) CanInterface() bool {
	panic("not implemented")
}

func (v Value) CanSet() bool {
	panic("not implemented")
}

func (v Value) CanUint() bool {
	panic("not implemented")
}

func (v Value) Cap() int {
	panic("not implemented")
}

func (v Value) Close() {
	panic("not implemented")
}

func (v Value) Comparable() bool {
	panic("not implemented")
}

func (v Value) Complex() complex128 {
	panic("not implemented")
}

func (v Value) Convert(t Type) Value {
	panic("not implemented")
}

func (v Value) Elem() Value {
	panic("not implemented")
}

func (v Value) Equal(u Value) bool {
	panic("not implemented")
}

func (v Value) Field(i int) Value {
	panic("not implemented")
}

func (v Value) FieldByIndex(index []int) Value {
	panic("not implemented")
}

func (v Value) FieldByIndexErr(index []int) (Value, error) {
	panic("not implemented")
}

func (v Value) FieldByName(name string) Value {
	panic("not implemented")
}

func (v Value) FieldByNameFunc(match func(string) bool) Value {
	panic("not implemented")
}

func (v Value) Float() float64 {
	panic("not implemented")
}

func (v Value) Grow(n int) {
	panic("not implemented")
}

func (v Value) Index(i int) Value {
	panic("not implemented")
}

func (v Value) Int() int64 {
	panic("not implemented")
}

func (v Value) Interface() (i any) {
	panic("not implemented")
}

func (v Value) IsNil() bool {
	panic("not implemented")
}

func (v Value) IsValid() bool {
	panic("not implemented")
}

func (v Value) IsZero() bool {
	panic("not implemented")
}

func (v Value) Kind() Kind {
	panic("not implemented")
}

func (v Value) Len() int {
	panic("not implemented")
}

func (v Value) MapIndex(key Value) Value {
	panic("not implemented")
}

func (v Value) MapKeys() []Value {
	panic("not implemented")
}

func (v Value) MapRange() *MapIter {
	panic("not implemented")
}

func (v Value) Method(i int) Value {
	panic("not implemented")
}

func (v Value) MethodByName(name string) Value {
	panic("not implemented")
}

func (v Value) NumField() int {
	panic("not implemented")
}

func (v Value) NumMethod() int {
	panic("not implemented")
}

func (v Value) OverflowComplex(x complex128) bool {
	panic("not implemented")
}

func (v Value) OverflowFloat(x float64) bool {
	panic("not implemented")
}

func (v Value) OverflowInt(x int64) bool {
	panic("not implemented")
}

func (v Value) OverflowUint(x uint64) bool {
	panic("not implemented")
}

func (v Value) Pointer() uintptr {
	panic("not implemented")
}

func (v Value) Recv() (x Value, ok bool) {
	panic("not implemented")
}

func (v Value) Send(x Value) {
	panic("not implemented")
}

func (v Value) Set(x Value) {
	panic("not implemented")
}

func (v Value) SetBool(x bool) {
	panic("not implemented")
}

func (v Value) SetBytes(x []byte) {
	panic("not implemented")
}

func (v Value) SetCap(n int) {
	panic("not implemented")
}

func (v Value) SetComplex(x complex128) {
	panic("not implemented")
}

func (v Value) SetFloat(x float64) {
	panic("not implemented")
}

func (v Value) SetInt(x int64) {
	panic("not implemented")
}

func (v Value) SetIterKey(iter *MapIter) {
	panic("not implemented")
}

func (v Value) SetIterValue(iter *MapIter) {
	panic("not implemented")
}

func (v Value) SetLen(n int) {
	panic("not implemented")
}

func (v Value) SetMapIndex(key, elem Value) {
	panic("not implemented")
}

func (v Value) SetPointer(x unsafe.Pointer) {
	panic("not implemented")
}

func (v Value) SetString(x string) {
	panic("not implemented")
}

func (v Value) SetUint(x uint64) {
	panic("not implemented")
}

func (v Value) SetZero() {
	panic("not implemented")
}

func (v Value) Slice(i, j int) Value {
	panic("not implemented")
}

func (v Value) Slice3(i, j, k int) Value {
	panic("not implemented")
}

func (v Value) String() string {
	panic("not implemented")
}

func (v Value) TryRecv() (x Value, ok bool) {
	panic("not implemented")
}

func (v Value) TrySend(x Value) bool {
	panic("not implemented")
}

func (v Value) Type() Type {
	panic("not implemented")
}

func (v Value) Uint() uint64 {
	panic("not implemented")
}

func (v Value) UnsafeAddr() uintptr {
	panic("not implemented")
}

func (v Value) UnsafePointer() unsafe.Pointer {
	panic("not implemented")
}

type ValueError struct {
	Method string
	Kind   Kind
}

func (e *ValueError) Error() string {
	panic("not implemented")
}
