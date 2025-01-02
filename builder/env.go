package builder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Env map[string]string

func Environment() (Env, error) {
	compilerPath, err := os.Executable()
	if err != nil {
		return nil, err
	}

	// The default root should be back one directory from the compiler
	sigoRoot, err := filepath.Abs(filepath.Join(filepath.Dir(compilerPath), ".."))
	if err != nil {
		return nil, err
	}

	// Get GOROOT from the system Go
	goRoot := getgoenv("GOROOT", "")
	goProxy := getgoenv("GOPROXY", "")
	goModCache := getgoenv("GOMODCACHE", "")

	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	// Get the user cache directory
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		// Use the temporary directory of the current operating system.
		cacheDir = os.TempDir()
	}

	// Return the environment
	return map[string]string{
		"SIGOROOT":  getenv("SIGOROOT", sigoRoot),
		"SIGOPATH":  getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"SIGOCACHE": getenv("SIGOCACHE", getenv("GOCACHE", filepath.Join(cacheDir, "go-build"))),

		"GOROOT":     getenv("GOROOT", goRoot),
		"GOPROXY":    getenv("GOROOT", goProxy),
		"GOMODCACHE": getenv("GOMODCACHE", goModCache),
		//"GOPATH":   getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"GOCACHE":  getenv("SIGOCACHE", getenv("GOCACHE", filepath.Join(cacheDir, "go-build"))),
		"GOTMPDIR": getenv("SIGOTMPDIR", filepath.Join(cacheDir, "sigo")),
		//"PATH":     os.Getenv("PATH"),
		"CC":      getenv("CC", ""),
		"LD":      getenv("LD", ""),
		"OBJCOPY": getenv("OBJCOPY", ""),
	}, nil
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

func getgoenv(key, _default string) (value string) {
	cmd := exec.Command("go", "env", key)
	output, err := cmd.Output()
	if err != nil {
		panic(err)
	}
	value = strings.TrimSpace(string(output))
	if len(value) == 0 {
		value = _default
	}
	return
}
