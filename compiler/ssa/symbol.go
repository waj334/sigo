package ssa

import "sync"

type SymbolInfo struct {
	LinkName        string
	Exported        bool
	ExternalLinkage bool
	IsCGoFunc       bool
	IsInterrupt     bool
	IsRequired      bool
	Linkage         string
	Attributes      map[string]struct{}
	Section         string
	Alignment       int64
	EmbedPatterns   []string

	// StackSize, when non-zero, overrides the default goroutine/coroutine
	// stack size for callers that launch this function via `go` or
	// `runtime.newcoro`. Set by `//sigo:stacksize N` on the function
	// declaration.
	StackSize uint64
}

type SymbolInfoStore struct {
	info map[string]*SymbolInfo
	mu   sync.Mutex
}

func NewSymbolInfoStore() *SymbolInfoStore {
	return &SymbolInfoStore{
		info: map[string]*SymbolInfo{},
	}
}

func (s *SymbolInfoStore) GetSymbolInfo(symbol string) *SymbolInfo {
	s.mu.Lock()
	defer s.mu.Unlock()

	info, ok := s.info[symbol]
	if !ok {
		info = &SymbolInfo{
			Attributes: map[string]struct{}{},
		}
		s.info[symbol] = info
	}
	return info
}
