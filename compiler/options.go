package compiler

import "strings"

type Verbosity int

const (
	Quiet Verbosity = iota
	Info
	Warning
	Verbose
	Debug
)

type SymbolInfo struct {
	LinkName        string
	Exported        bool
	ExternalLinkage bool
	IsInterrupt     bool
}

type Options struct {
	Target             *Target
	Symbols            map[string]*SymbolInfo
	GenerateDebugInfo  bool
	Verbosity          Verbosity
	PathMappings       map[string]string
	GoroutineStackSize uint64
	PrimitivesAsCTypes bool
}

func NewOptions() *Options {
	return &Options{
		Symbols:            map[string]*SymbolInfo{},
		PathMappings:       map[string]string{},
		GoroutineStackSize: 2000,
	}
}

func (o *Options) GetSymbolInfo(symbol string) *SymbolInfo {
	info, ok := o.Symbols[symbol]
	if !ok {
		info = &SymbolInfo{}
		o.Symbols[symbol] = info
	}
	return info
}

func (o *Options) MapPath(path string) string {
	for from, to := range o.PathMappings {
		if strings.Contains(path, from) {
			return strings.Replace(path, from, to, -1)
		}
	}
	return path
}
