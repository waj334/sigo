package atdf

import (
	"pkg.si-go.dev/sigo/cmd/tbdef-gen/common"
)

type Offsetable interface {
	Offset() common.Address
}

type ATDF struct {
	Devices DevicesElement   `xml:"devices"`
	Modules ModulesElement   `xml:"modules"`
	Pinouts []PinoutsElement `xml:"pinouts"`
}

type DevicesElement struct {
	Elements []DeviceElement `xml:"device"`
}

type DeviceElement struct {
	Name           string                `xml:"name,attr"`
	Architecture   string                `xml:"architecture,attr"`
	Family         string                `xml:"family,attr"`
	Series         string                `xml:"series,attr"`
	AddressSpaces  AddressSpacesElement  `xml:"address-spaces"`
	Parameters     ParametersElement     `xml:"parameters"`
	Peripherals    PeripheralsElement    `xml:"peripherals"`
	Interrupts     InterruptsElement     `xml:"interrupts"`
	Events         EventsElement         `xml:"events"`
	Interfaces     InterfacesElement     `xml:"interfaces"`
	PropertyGroups PropertyGroupElements `xml:"property-groups"`
}

type AddressSpacesElement struct {
	Elements []AddressSpaceElement `xml:"address-space"`
}

type AddressSpaceElement struct {
	Id             string                 `xml:"id,attr"`
	Name           string                 `xml:"name,attr"`
	Start          common.Address         `xml:"start,attr"`
	Size           common.Address         `xml:"size,attr"`
	Endianness     string                 `xml:"endianness,attr"`
	MemorySegments []MemorySegmentElement `xml:"memory-segment"`
}

type MemorySegmentElement struct {
	Name     string         `xml:"name,attr"`
	Start    common.Address `xml:"start,attr"`
	Size     common.Address `xml:"size,attr"`
	Type     string         `xml:"type,attr"`
	PageSize common.Address `xml:"pagesize,attr"`
	RW       string         `xml:"rw,attr"`
	Exec     bool           `xml:"exec,attr,omitempty"`
}

type ParametersElement struct {
	Elements []ParameterElement `xml:"param"`
}

type ParameterElement struct {
	Name    string         `xml:"name,attr"`
	Value   common.Address `xml:"value,attr"`
	Caption string         `xml:"caption,attr,omitempty"`
}

type PeripheralsElement struct {
	Modules []ModuleGroupElement `xml:"module"`
}

type ModuleGroupElement struct {
	Name      string            `xml:"name,attr"`
	Id        string            `xml:"id,attr"`
	Version   string            `xml:"version,attr"`
	Instances []InstanceElement `xml:"instance"`
}

type InstanceElement struct {
	Name           string                         `xml:"name,attr"`
	RegisterGroups []InstanceRegisterGroupElement `xml:"register-group"`
	Signals        SignalsElement                 `xml:"signals"`
	Parameters     ParametersElement              `xml:"parameters"`
}

type InstanceRegisterGroupElement struct {
	Name         string         `xml:"name,attr"`
	NameInModule string         `xml:"name-in-module,attr"`
	AddressSpace string         `xml:"address-space,attr"`
	Offset       common.Address `xml:"offset,attr"`
}

type SignalsElement struct {
	Elements SignalElement `xml:"signal"`
}

type SignalElement struct {
	Group    string `xml:"group,attr"`
	Index    int    `xml:"index,attr"`
	Function string `xml:"function,attr"`
	Pad      string `xml:"pad,attr"`
}

type InterruptsElement struct {
	Elements []InterruptElement `xml:"interrupt"`
}

type InterruptElement struct {
	Name             string `xml:"name,attr"`
	Index            int    `xml:"index,attr"`
	Caption          string `xml:"caption,attr,omitempty"`
	AlternateCaption string `xml:"alternate-caption,attr"`
}

type EventsElement struct {
	Generators []GeneratorElement `xml:"generators"`
}

type GeneratorElement struct {
	Name           string         `xml:"name,attr"`
	Index          common.Address `xml:"index,attr"`
	ModuleInstance string         `xml:"module-instance,attr"`
}

type InterfacesElement struct {
	Elements InterfaceElement `xml:"interface"`
}

type InterfaceElement struct {
	Name string `xml:"name,attr"`
	Type string `xml:"type,attr"`
}

type PropertyGroupElements struct {
	Elements []PropertyElement `xml:"property"`
}

type PropertyElement struct {
	Name  string         `xml:"name,attr"`
	Value common.Address `xml:"value,attr"`
}

type ModulesElement struct {
	Elements []ModuleElement `xml:"module"`
}

type ModuleElement struct {
	Name           string                       `xml:"name,attr"`
	Id             string                       `xml:"id,attr"`
	Version        string                       `xml:"version,attr"`
	Caption        string                       `xml:"caption,attr,omitempty"`
	RegisterGroups []ModuleRegisterGroupElement `xml:"register-group"`
	ValueGroups    []ModuleValueGroupElement    `xml:"value-group"`
}

func (m *ModuleElement) FindValueGroup(name string) *ModuleValueGroupElement {
	for _, group := range m.ValueGroups {
		if group.Name == name {
			return &group
		}
	}
	return nil
}

type ModuleRegisterGroupElement struct {
	Name         string                       `xml:"name,attr"`
	NameInModule string                       `xml:"name-in-module,attr"`
	Size         common.Address               `xml:"size,attr"`
	Caption      string                       `xml:"caption,attr,omitempty"`
	Count        common.Address               `xml:"count,attr"`
	Offset_      common.Address               `xml:"offset,attr"`
	Registers    []RegisterElement            `xml:"register"`
	Groups       []ModuleRegisterGroupElement `xml:"register-group"`
	Modes        []ModeElement                `xml:"mode"`
}

func (m ModuleRegisterGroupElement) Offset() common.Address {
	return m.Offset_
}

type RegisterElement struct {
	Name      string            `xml:"name,attr"`
	Mode      string            `xml:"modes,attr"`
	Modes     []ModeElement     `xml:"mode"`
	Offset_   common.Address    `xml:"offset,attr"`
	RW        string            `xml:"rw,attr"`
	Access    string            `xml:"access,omitempty"`
	Size      common.Address    `xml:"size,attr"`
	Count     common.Address    `xml:"count,attr"`
	InitValue common.Address    `xml:"initval,attr"`
	Caption   string            `xml:"caption,attr,omitempty"`
	BitFields []BitFieldElement `xml:"bitfield"`
}

func (r RegisterElement) Offset() common.Address {
	return r.Offset_
}

type BitFieldElement struct {
	Name    string         `xml:"name,attr"`
	Modes   string         `xml:"modes,attr"`
	Caption string         `xml:"caption,attr,omitempty"`
	Mask    common.Address `xml:"mask,attr"`
	Values  *string        `xml:"values,attr,omitempty"`
}

type ModuleValueGroupElement struct {
	Name     string               `xml:"name,attr"`
	Elements []ModuleValueElement `xml:"value"`
}

type ModuleValueElement struct {
	Name    string         `xml:"name,attr"`
	Caption string         `xml:"caption,attr,omitempty"`
	Value   common.Address `xml:"value,attr"`
}

type PinoutsElement struct {
	Pinout []PinoutElement `xml:"pinout"`
}

type PinoutElement struct {
	Pins []PinElement `xml:"pin"`
}

type PinElement struct {
	Position string `xml:"position,attr"`
	Pad      string `xml:"pad,attr"`
}

type ModeElement struct {
	Name      string         `xml:"name,attr"`
	Qualifier string         `xml:"qualifier,attr"`
	Value     common.Address `xml:"value,attr"`
	Caption   string         `xml:"caption,attr,omitempty"`
}
