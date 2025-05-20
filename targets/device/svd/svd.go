package svd

import (
	"pkg.si-go.dev/sigo/targets/device"
)

type Addressable interface {
	GetAddressOffset() device.Address
}

type DeviceElement struct {
	Name             string             `xml:"name"`
	Description      string             `xml:"description"`
	Series           *string            `xml:"series"`
	Version          string             `xml:"version"`
	Vendor           *string            `xml:"vendor"`
	VendorId         *string            `xml:"vendorId"`
	CPU              *CPUElement        `xml:"cpu"`
	AddressableWidth device.Address     `xml:"addressUnitBits"`
	BitWidth         device.Address     `xml:"width"`
	RegisterSize     *device.Address    `xml:"size"`
	DefaultAccess    *string            `xml:"access"`
	ResetValue       *device.Address    `xml:"resetValue"`
	ResetMask        *device.Address    `xml:"resetMask"`
	Peripherals      PeripheralsElement `xml:"peripherals"`
}

type CPUElement struct {
	Name                string         `xml:"name"`
	Revision            string         `xml:"revision"`
	Endian              string         `xml:"endian"`
	MPUPresent          string         `xml:"mpuPresent"`
	FPUPresent          string         `xml:"fpuPresent"`
	NVICPriorityBits    device.Address `xml:"nvicPrioBits"`
	VendorSystickConfig bool           `xml:"vendorSystickConfig"`
}

type PeripheralsElement struct {
	Elements []PeripheralElement `xml:"peripheral"`
}

func (p PeripheralsElement) Find(name string) (int, bool) {
	if len(name) > 0 {
		for i, pp := range p.Elements {
			if pp.Name == name {
				return i, true
			}
		}
	}
	return -1, false
}

type PeripheralElement struct {
	Name         string               `xml:"name"`
	Description  *string              `xml:"description"`
	Group        *string              `xml:"groupName"`
	Access       *string              `xml:"access"`
	BaseAddress  device.Address       `xml:"baseAddress"`
	AddressBlock *AddressBlockElement `xml:"addressBlock"`
	Interrupts   *[]InterruptElement  `xml:"interrupt"`
	Registers    *RegistersElement    `xml:"registers"`
	DerivedFrom  *string              `xml:"derivedFrom,attr"`
}

type AddressBlockElement struct {
	Offset device.Address `xml:"offset"`
	Size   device.Address `xml:"size"`
}

type InterruptElement struct {
	Name        string `xml:"name"`
	Description string `xml:"description"`
	Value       int    `xml:"value"`
}

type RegistersElement struct {
	RegisterElements []RegisterElement `xml:"register"`
	ClusterElements  []ClusterElement  `xml:"cluster"`
}

type ClusterElement struct {
	Name          string            `xml:"name"`
	Description   string            `xml:"description"`
	Count         device.Address    `xml:"dim"`
	Increment     device.Address    `xml:"dimIncrement"`
	AddressOffset device.Address    `xml:"addressOffset"`
	Registers     []RegisterElement `xml:"register"`
}

func (c ClusterElement) GetAddressOffset() device.Address {
	return c.AddressOffset
}

type RegisterElement struct {
	Name          string         `xml:"name"`
	DisplayName   string         `xml:"displayName"`
	Description   string         `xml:"description"`
	AddressOffset device.Address `xml:"addressOffset"`
	Size          device.Address `xml:"size"`
	Fields        FieldElements  `xml:"fields"`
	Count         device.Address `xml:"dim"`
	Increment     device.Address `xml:"dimIncrement"`
	Access        string         `xml:"access"`
	Alternative   string         `xml:"alternateRegister"`
}

func (r RegisterElement) GetAddressOffset() device.Address {
	return r.AddressOffset
}

type FieldElements struct {
	Elements []FieldElement `xml:"field"`
}

type FieldElement struct {
	Name             string                  `xml:"name"`
	Description      string                  `xml:"description"`
	BitOffset        device.Address          `xml:"bitOffset"`
	BitWidth         device.Address          `xml:"bitWidth"`
	BitRange         string                  `xml:"bitRange"`
	Access           string                  `xml:"access"`
	EnumeratedValues EnumeratedValuesElement `xml:"enumeratedValues"`
}

type EnumeratedValuesElement struct {
	Name     string                   `xml:"name"`
	Elements []EnumeratedValueElement `xml:"enumeratedValue"`
}

type EnumeratedValueElement struct {
	Name        string         `xml:"name"`
	Description string         `xml:"description"`
	Value       device.Address `xml:"value"`
}
