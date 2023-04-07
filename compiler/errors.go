package compiler

import "errors"

var (
	ErrUnknownType               = errors.New("unknown type")
	ErrTargetMissingArchitecture = errors.New("target is missing architecture type")
	ErrTargetMissingCpu          = errors.New("target is missing CPU type")
	ErrTargetMissingTriple       = errors.New("target is missing target triple value")
	ErrTargetInformationFailed   = errors.New("failed to get target information")
)
