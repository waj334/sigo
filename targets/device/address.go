package device

import (
	"encoding/xml"
	"fmt"
	"strconv"
	"strings"
)

type Address uintptr

func (a *Address) UnmarshalXMLAttr(attr xml.Attr) error {
	return a.decode(attr.Value)
}

func (a *Address) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var v string
	err := d.DecodeElement(&v, &start)
	if err != nil {
		return err
	}
	return a.decode(v)
}

func (a *Address) UnmarshalJSON(bytes []byte) error {
	return a.decode(string(bytes))
}

func (a *Address) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf("\"%#x\"", *a)), nil
}

func (a *Address) decode(input string) error {
	var value uint64
	var err error

	input = strings.ToLower(input)
	input = strings.Trim(input, "\"")

	if strings.Contains(input, "0x") {
		input = strings.TrimPrefix(input, "0x")
		value, err = strconv.ParseUint(input, 16, 64)
	} else {
		value, err = strconv.ParseUint(input, 10, 64)
	}

	if err != nil {
		return err
	}

	*a = Address(value)
	return nil
}
