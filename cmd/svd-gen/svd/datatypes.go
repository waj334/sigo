package svd

import (
	"encoding/xml"
	"strconv"
	"strings"
)

type Integer uint64

func (h *Integer) UnmarshalXML(d *xml.Decoder, start xml.StartElement) (err error) {
	var v string
	d.DecodeElement(&v, &start)

	var value uint64
	if strings.Contains(v, "0x") {
		s := strings.TrimPrefix(v, "0x")
		value, err = strconv.ParseUint(s, 16, 64)
	} else {
		value, err = strconv.ParseUint(v, 10, 64)
	}

	if err != nil {
		return err
	}
	*h = Integer(value)
	return nil
}
