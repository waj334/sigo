package types

import (
	"encoding/xml"
	"strconv"
	"strings"
)

type Integer int64

func (i *Integer) UnmarshalXML(d *xml.Decoder, start xml.StartElement) (err error) {
	var v string
	d.DecodeElement(&v, &start)

	var value int64
	if strings.Contains(v, "0x") {
		s := strings.TrimPrefix(v, "0x")
		value, err = strconv.ParseInt(s, 16, 64)
	} else {
		value, err = strconv.ParseInt(v, 10, 64)
	}

	if err != nil {
		return err
	}
	*i = Integer(value)
	return nil
}

func (i *Integer) UnmarshalXMLAttr(attr xml.Attr) (err error) {
	var value int64
	strVal := strings.ReplaceAll(attr.Value, "X", "x")
	if strings.Contains(strVal, "0x") {
		strVal = strings.TrimPrefix(strVal, "0x")
		value, err = strconv.ParseInt(strVal, 16, 64)
	} else {
		value, err = strconv.ParseInt(strVal, 10, 64)
	}
	if err != nil {
		return err
	}
	*i = Integer(value)
	return nil
}

type UInteger int64

func (i *UInteger) UnmarshalXML(d *xml.Decoder, start xml.StartElement) (err error) {
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
	*i = UInteger(value)
	return nil
}

func (i *UInteger) UnmarshalXMLAttr(attr xml.Attr) (err error) {
	var value uint64
	strVal := strings.ReplaceAll(attr.Value, "X", "x")
	if strings.Contains(strVal, "0x") {
		strVal := strings.TrimPrefix(strVal, "0x")
		value, err = strconv.ParseUint(strVal, 16, 64)
	} else {
		value, err = strconv.ParseUint(strVal, 10, 64)
	}
	if err != nil {
		return err
	}
	*i = UInteger(value)
	return nil
}
