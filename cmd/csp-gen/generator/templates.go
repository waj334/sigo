package generator

import (
	"fmt"
	"github.com/sirkon/go-format/v2"
)

const (
	boolGetter = `func (field *${type}) Get${identifier}() bool {
	return volatile.LoadUint${width}((*uint${width})(field))&(1<<${offset}) != 0
}`

	boolSetter = `func (field *${type}) Set${identifier}(enable bool) {
	if enable {
		volatile.StoreUint${width}((*uint${width})(field), volatile.LoadUint${width}((*uint${width})(field))|(1<<${offset}))
	} else {
		volatile.StoreUint${width}((*uint${width})(field), volatile.LoadUint${width}((*uint${width})(field))&^(1<<${offset}))
	}
}`

	intGetter = `func (field *${type}) Get${identifier}() ${return} {
	return ${return}(volatile.LoadUint${width}((*uint32)(field))&${mask}) >> ${offset}
}`

	intSetter = `func (field *${type}) Set${identifier}(value ${return}) {
	volatile.StoreUint${width}((*uint${width})(field), (volatile.LoadUint${width}((*uint${width})(field))&^${mask})|(uint${width}(value)<<${offset}))
}`
)

func fieldParams(f Field) format.Values {
	var returnType string
	if len(f.Constants.Values) > 0 {
		returnType = f.Constants.TypeName()
	} else {
		returnType = fmt.Sprintf("uint%d", NextPow2(f.Width))
	}

	return format.Values{
		"type":       f.TypeName(),
		"return":     returnType,
		"identifier": f.Identifier,
		"width":      f.Register.Width,
		"mask":       fmt.Sprintf("%#x", Mask(f.Width, f.Offset)),
		"offset":     f.Offset,
	}
}
