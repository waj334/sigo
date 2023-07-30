package svd

import "omibyte.io/sigo/cmd/csp-gen/types"

type Addressable interface {
	GetAddressOffset() types.Integer
}

type DeviceElement struct {
	Name             string             `xml:"name"`
	Description      string             `xml:"description"`
	Series           string             `xml:"series"`
	Version          string             `xml:"version"`
	Vendor           string             `xml:"vendor"`
	VendorId         string             `xml:"vendorId"`
	CPU              CPUElement         `xml:"cpu"`
	AddressableWidth types.Integer      `xml:"addressUnitBits"`
	BitWidth         types.Integer      `xml:"width"`
	RegisterSize     types.Integer      `xml:"size"`
	DefaultAccess    string             `xml:"access"`
	ResetValue       types.Integer      `xml:"resetValue"`
	ResetMask        types.Integer      `xml:"resetMask"`
	Peripherals      PeripheralsElement `xml:"peripherals"`
}

type CPUElement struct {
	Name                string        `xml:"name"`
	Revision            string        `xml:"revision"`
	Endian              string        `xml:"endian"`
	MPUPresent          string        `xml:"mpuPresent"`
	FPUPresent          string        `xml:"fpuPresent"`
	NVICPriorityBits    types.Integer `xml:"nvicPrioBits"`
	VendorSystickConfig bool          `xml:"vendorSystickConfig"`
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
	Name         string              `xml:"name"`
	Description  string              `xml:"description"`
	Group        string              `xml:"groupName"`
	BaseAddress  types.Integer       `xml:"baseAddress"`
	AddressBlock AddressBlockElement `xml:"addressBlock"`
	Interrupts   []InterruptElement  `xml:"interrupt"`
	Registers    RegistersElement    `xml:"registers"`
	DerivedFrom  string              `xml:"derivedFrom,attr"`
}

type AddressBlockElement struct {
	Offset types.Integer `xml:"offset"`
	Size   types.Integer `xml:"size"`
}

type InterruptElement struct {
	Name        string        `xml:"name"`
	Description string        `xml:"description"`
	Value       types.Integer `xml:"value"`
}

type RegistersElement struct {
	RegisterElements []RegisterElement `xml:"register"`
	ClusterElements  []ClusterElement  `xml:"cluster"`
}

type ClusterElement struct {
	Name          string            `xml:"name"`
	Description   string            `xml:"description"`
	Count         types.Integer     `xml:"dim"`
	Increment     types.Integer     `xml:"dimIncrement"`
	AddressOffset types.Integer     `xml:"addressOffset"`
	Registers     []RegisterElement `xml:"register"`
}

func (c ClusterElement) GetAddressOffset() types.Integer {
	return c.AddressOffset
}

type RegisterElement struct {
	Name          string        `xml:"name"`
	Description   string        `xml:"description"`
	AddressOffset types.Integer `xml:"addressOffset"`
	Size          types.Integer `xml:"size"`
	Fields        FieldElements `xml:"fields"`
	Count         types.Integer `xml:"dim"`
	Increment     types.Integer `xml:"dimIncrement"`
	Access        string        `xml:"access"`
	Alternative   string        `xml:"alternateRegister"`
}

func (r RegisterElement) GetAddressOffset() types.Integer {
	return r.AddressOffset
}

type FieldElements struct {
	Elements []FieldElement `xml:"field"`
}

type FieldElement struct {
	Name             string                  `xml:"name"`
	Description      string                  `xml:"description"`
	BitOffset        types.Integer           `xml:"bitOffset"`
	BitWidth         types.Integer           `xml:"bitWidth"`
	Access           string                  `xml:"access"`
	EnumeratedValues EnumeratedValuesElement `xml:"enumeratedValues"`
}

type EnumeratedValuesElement struct {
	Name     string                   `xml:"name"`
	Elements []EnumeratedValueElement `xml:"enumeratedValue"`
}

type EnumeratedValueElement struct {
	Name        string        `xml:"name"`
	Description string        `xml:"description"`
	Value       types.Integer `xml:"value"`
}
