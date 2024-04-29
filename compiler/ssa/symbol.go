package ssa

import "sync"

type SymbolInfo struct {
	LinkName        string
	Exported        bool
	ExternalLinkage bool
	IsInterrupt     bool
	IsRequired      bool
	Linkage         string
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
		info = &SymbolInfo{}
		s.info[symbol] = info
	}
	return info
}
