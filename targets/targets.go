package targets

import (
	_ "embed"
	"errors"
	"strings"

	"golang.org/x/exp/slices"
	"gopkg.in/yaml.v3"

	"omibyte.io/sigo/llvm"
)

//go:embed targets.yaml
var rawTargets []byte

var targets Targets
var ErrTargetInformationFailed = errors.New("failed to get target information")

func All() Targets {
	return targets
}

type Targets []TargetInfo
type TargetInfo struct {
	Series       string   `yaml:"series"`
	Chips        []string `yaml:"chips"`
	ChipPackage  string   `yaml:"chipPackage"`
	Cpu          string   `yaml:"cpu"`
	Architecture string   `yaml:"architecture"`
	Alignment    int      `yaml:"alignment"`
	Triple       string   `yaml:"triple"`
	Tags         []string `yaml:"tags"`
	Features     []string `yaml:"features"`
	Float        string   `yaml:"float"`
}

func (t TargetInfo) FormatFeatureString() string {
	features := make([]string, len(t.Features))
	for i, feature := range t.Features {
		features[i] = "+" + feature
	}
	return strings.Join(features, ",")
}

func (t TargetInfo) CreateTarget() (llvm.LLVMTargetRef, error) {
	// Get the target from the triple
	target, errMsg, ok := llvm.GetTargetFromTriple(t.Triple)
	if !ok {
		if len(errMsg) > 0 {
			return llvm.LLVMTargetRef{}, errors.Join(ErrTargetInformationFailed, errors.New(errMsg))
		}
		return llvm.LLVMTargetRef{}, ErrTargetInformationFailed
	}
	return target, nil
}

func (t TargetInfo) CreateTargetMachine(target llvm.LLVMTargetRef) llvm.LLVMTargetMachineRef {
	// Create the target machine with the desired CPU and features
	machineRef := llvm.CreateTargetMachine(
		target,
		t.Triple,
		t.Cpu,
		t.FormatFeatureString(),
		llvm.LLVMCodeGenOptLevel(llvm.CodeGenLevelNone),
		llvm.LLVMRelocMode(llvm.RelocDefault),
		llvm.LLVMCodeModel(llvm.CodeModelDefault))
	return machineRef
}

func (t Targets) FindBySeries(name string) (TargetInfo, error) {
	for _, target := range t {
		if target.Series == strings.ToLower(name) {
			return target, nil
		}
	}
	return TargetInfo{}, errors.New("series not found")
}

func (t Targets) FindByChip(name string) (TargetInfo, error) {
	for _, target := range t {
		if slices.Contains(target.Chips, strings.ToLower(name)) {
			return target, nil
		}
	}
	return TargetInfo{}, errors.New("series not found")
}

func init() {
	var t struct {
		Elements []TargetInfo `yaml:"targets"`
	}
	if err := yaml.Unmarshal(rawTargets, &t); err != nil {
		panic(err)
	}

	targets = t.Elements
}
