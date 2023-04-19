package builder

import (
	"fmt"
	"os"
	"path/filepath"
)

type Env map[string]string

func Environment() Env {
	compilerPath, err := os.Executable()
	if err != nil {
		panic(err)
	}

	// The default root should be back one directory from the compiler
	defaultRoot, err := filepath.Abs(filepath.Dir(compilerPath) + "/..")
	if err != nil {
		panic(err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	// Return the environment
	return map[string]string{
		"SIGOROOT":  getenv("SIGOROOT", defaultRoot),
		"SIGOPATH":  getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"SIGOCACHE": getenv("SIGOCACHE", getenv("GOCACHE", "C:\\Users\\waj33\\AppData\\Local\\go-build")),

		"GOROOT":   getenv("GOROOT", defaultRoot),
		"GOPATH":   getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"GOCACHE":  getenv("SIGOCACHE", getenv("GOCACHE", "C:\\Users\\waj33\\AppData\\Local\\go-build")),
		"GOTMPDIR": getenv("SIGOTMPDIR", filepath.Join(os.TempDir(), "/sigo")),
		//"PATH":     os.Getenv("PATH"),
	}
}

func (e Env) Print() {
	for k, v := range e {
		fmt.Printf("set %s=%s", k, v)
	}
}

func (e Env) Value(key string) string {
	if v, ok := e[key]; ok {
		return v
	}
	return ""
}

func (e Env) List() []string {
	var result []string
	for key, value := range e {
		result = append(result, fmt.Sprintf("%s=%s", key, value))
	}
	return result
}

func getenv(key, _default string) (value string) {
	value = os.Getenv(key)
	if len(value) == 0 {
		value = _default
	}
	return value
}
