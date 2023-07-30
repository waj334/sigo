package compiler

import (
	"strings"
	"sync"
)

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
	Linkage         string
}

type Options struct {
	Target             *Target
	Symbols            map[string]*SymbolInfo
	GenerateDebugInfo  bool
	Verbosity          Verbosity
	PathMappings       map[string]string
	GoroutineStackSize uint64
	MaxStackSize       uint64
	PrimitivesAsCTypes bool
	mu                 sync.Mutex
}

func NewOptions() *Options {
	return &Options{
		Symbols:            map[string]*SymbolInfo{},
		PathMappings:       map[string]string{},
		GoroutineStackSize: 2048,
		MaxStackSize:       256,
	}
}

func (o *Options) WithTarget(target *Target) *Options {
	o.mu.Lock()
	defer o.mu.Unlock()

	sym := map[string]*SymbolInfo{}
	for k, v := range o.Symbols {
		sym[k] = v
	}

	return &Options{
		Target:             target,
		Symbols:            sym,
		GenerateDebugInfo:  o.GenerateDebugInfo,
		Verbosity:          o.Verbosity,
		PathMappings:       o.PathMappings,
		GoroutineStackSize: o.GoroutineStackSize,
		PrimitivesAsCTypes: o.PrimitivesAsCTypes,
		mu:                 sync.Mutex{},
		MaxStackSize:       o.MaxStackSize,
	}
}

func (o *Options) GetSymbolInfo(symbol string) *SymbolInfo {
	o.mu.Lock()
	defer o.mu.Unlock()

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
