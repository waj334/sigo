package builder

import (
	_ "embed"
	"errors"
	"golang.org/x/exp/slices"
	"gopkg.in/yaml.v3"
	"omibyte.io/sigo/compiler"
	"strings"
)

//go:embed targets.yaml
var rawTargets []byte
var targets Targets

type Targets []compiler.TargetInfo

func (t Targets) FindBySeries(name string) (compiler.TargetInfo, error) {
	for _, target := range t {
		if target.Series == strings.ToLower(name) {
			return target, nil
		}
	}
	return compiler.TargetInfo{}, errors.New("series not found")
}

func (t Targets) FindByChip(name string) (compiler.TargetInfo, error) {
	for _, target := range t {
		if slices.Contains(target.Chips, strings.ToLower(name)) {
			return target, nil
		}
	}
	return compiler.TargetInfo{}, errors.New("series not found")
}

func init() {
	var t struct {
		Elements []compiler.TargetInfo `yaml:"targets"`
	}
	if err := yaml.Unmarshal(rawTargets, &t); err != nil {
		panic(err)
	}

	targets = t.Elements
}
