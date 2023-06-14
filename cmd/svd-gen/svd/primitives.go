package svd

type Addressable interface {
	GetAddressOffset() Integer
}

type DeviceElement struct {
	Name             string             `xml:"name"`
	Description      string             `xml:"description"`
	Series           string             `xml:"series"`
	Version          string             `xml:"version"`
	Vendor           string             `xml:"vendor"`
	VendorId         string             `xml:"vendorId"`
	CPU              CPUElement         `xml:"cpu"`
	AddressableWidth Integer            `xml:"addressUnitBits"`
	BitWidth         Integer            `xml:"width"`
	RegisterSize     Integer            `xml:"size"`
	DefaultAccess    string             `xml:"access"`
	ResetValue       Integer            `xml:"resetValue"`
	ResetMask        Integer            `xml:"resetMask"`
	Peripherals      PeripheralsElement `xml:"peripherals"`
}

type CPUElement struct {
	Name                string  `xml:"name"`
	Revision            string  `xml:"revision"`
	Endian              string  `xml:"endian"`
	MPUPresent          string  `xml:"mpuPresent"`
	FPUPresent          string  `xml:"fpuPresent"`
	NVICPriorityBits    Integer `xml:"nvicPrioBits"`
	VendorSystickConfig bool    `xml:"vendorSystickConfig"`
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
	BaseAddress  Integer             `xml:"baseAddress"`
	AddressBlock AddressBlockElement `xml:"addressBlock"`
	Interrupts   []InterruptElement  `xml:"interrupt"`
	Registers    RegistersElement    `xml:"registers"`
	DerivedFrom  string              `xml:"derivedFrom,attr"`
}

type AddressBlockElement struct {
	Offset Integer `xml:"offset"`
	Size   Integer `xml:"size"`
}

type InterruptElement struct {
	Name        string  `xml:"name"`
	Description string  `xml:"description"`
	Value       Integer `xml:"value"`
}

type RegistersElement struct {
	RegisterElements []RegisterElement `xml:"register"`
	ClusterElements  []ClusterElement  `xml:"cluster"`
}

type ClusterElement struct {
	Name          string            `xml:"name"`
	Description   string            `xml:"description"`
	Count         Integer           `xml:"dim"`
	Increment     Integer           `xml:"dimIncrement"`
	AddressOffset Integer           `xml:"addressOffset"`
	Registers     []RegisterElement `xml:"register"`
}

func (c ClusterElement) GetAddressOffset() Integer {
	return c.AddressOffset
}

type RegisterElement struct {
	Name          string        `xml:"name"`
	Description   string        `xml:"description"`
	AddressOffset Integer       `xml:"addressOffset"`
	Size          Integer       `xml:"size"`
	Fields        FieldElements `xml:"fields"`
	Count         Integer       `xml:"dim"`
	Increment     Integer       `xml:"dimIncrement"`
	Access        string        `xml:"access"`
	Alternative   string        `xml:"alternateRegister"`
}

func (r RegisterElement) GetAddressOffset() Integer {
	return r.AddressOffset
}

type FieldElements struct {
	Elements []FieldElement `xml:"field"`
}

type FieldElement struct {
	Name             string                  `xml:"name"`
	Description      string                  `xml:"description"`
	BitOffset        Integer                 `xml:"bitOffset"`
	BitWidth         Integer                 `xml:"bitWidth"`
	Access           string                  `xml:"access"`
	EnumeratedValues EnumeratedValuesElement `xml:"enumeratedValues"`
}

type EnumeratedValuesElement struct {
	Name     string                   `xml:"name"`
	Elements []EnumeratedValueElement `xml:"enumeratedValue"`
}

type EnumeratedValueElement struct {
	Name        string  `xml:"name"`
	Description string  `xml:"description"`
	Value       Integer `xml:"value"`
}
