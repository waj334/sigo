package compiler

import (
	"errors"
	"omibyte.io/sigo/llvm"
	"strings"
)

type TargetInfo struct {
	Series       string   `yaml:"series"`
	Chips        []string `yaml:"chips"`
	ChipPackage  string   `yaml:"chipPackage"`
	Cpu          string   `yaml:"cpu"`
	Architecture string   `yaml:"architecture"`
	Triple       string   `yaml:"triple"`
	Tags         []string `yaml:"tags"`
	Features     []string `yaml:"features"`
	Float        string   `yaml:"float"`
}

type Target struct {
	architecture string
	cpu          string
	triple       string
	optimization string
	features     []string

	targetRef  llvm.LLVMTargetRef
	machineRef llvm.LLVMTargetMachineRef
	dataLayout llvm.LLVMTargetDataRef
}

func NewTargetFromMap(info map[string]string) (*Target, error) {
	var target Target

	target.architecture = info["architecture"]
	target.cpu = info["cpu"]
	target.triple = info["triple"]

	if features, ok := info["features"]; ok {
		target.features = strings.Split(features, ",")
	}

	// architecture, cpu and triple are required values
	if len(target.architecture) == 0 {
		return nil, ErrTargetMissingArchitecture
	}

	if len(target.cpu) == 0 {
		return nil, ErrTargetMissingCpu
	}

	if len(target.triple) == 0 {
		return nil, ErrTargetMissingTriple
	}

	return &target, nil
}

func NewTarget(info TargetInfo, additionalFeatures []string) (*Target, error) {
	var target Target

	target.architecture = info.Architecture
	target.cpu = info.Cpu
	target.triple = info.Triple
	target.features = append(info.Features, additionalFeatures...)

	// architecture, cpu and triple are required values
	if len(target.architecture) == 0 {
		return nil, ErrTargetMissingArchitecture
	}

	if len(target.cpu) == 0 {
		return nil, ErrTargetMissingCpu
	}

	if len(target.triple) == 0 {
		return nil, ErrTargetMissingTriple
	}

	return &target, nil
}

func (t *Target) Initialize() error {
	// Get the target from the triple
	target, errMsg, ok := llvm.GetTargetFromTriple(t.triple)
	if !ok {
		if len(errMsg) > 0 {
			return errors.Join(ErrTargetInformationFailed, errors.New(errMsg))
		}
		return ErrTargetInformationFailed
	}

	// Store the target ref
	t.targetRef = target

	// Determine the optimization level
	var optLevel llvm.LLVMCodeGenOptLevel
	switch t.optimization {
	default:
		// Disable optimizations
		optLevel = llvm.LLVMCodeGenOptLevel(llvm.CodeGenLevelNone)
	}

	// Create the target machine with the desired CPU and features
	t.machineRef = llvm.CreateTargetMachine(
		target,
		t.triple,
		t.cpu,
		t.featuresString(),
		optLevel,
		llvm.LLVMRelocMode(llvm.RelocDefault),
		llvm.LLVMCodeModel(llvm.CodeModelDefault))

	// Get the data layout
	t.dataLayout = llvm.CreateTargetDataLayout(t.machineRef)

	return nil
}

func (t *Target) Dispose() {
	llvm.DisposeTargetMachine(t.machineRef)
	llvm.DisposeTargetData(t.dataLayout)
}

func (t *Target) featuresString() string {
	features := make([]string, len(t.features))
	for i, feature := range t.features {
		features[i] = "+" + feature
	}
	return strings.Join(features, ",")
}

func (t *Target) Ref() llvm.LLVMTargetRef {
	return t.targetRef
}

func (t *Target) Machine() llvm.LLVMTargetMachineRef {
	return t.machineRef
}

func (t *Target) Triple() string {
	return t.triple
}

func (t *Target) Layout() llvm.LLVMTargetDataRef {
	return t.dataLayout
}
