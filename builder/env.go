package builder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

type Env map[string]string

func Environment() Env {
	compilerPath, err := os.Executable()
	if err != nil {
		panic(err)
	}

	// The default root should be back one directory from the compiler
	sigoRoot, err := filepath.Abs(filepath.Dir(compilerPath) + "/..")
	if err != nil {
		panic(err)
	}

	goRoot := sigoRoot

	// Attempt to get GOROOT from the system Go
	cmd := exec.Command("go", "env", "GOROOT")
	if output, err := cmd.Output(); err != nil {
		goRoot = string(output)
	}

	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	// Get the user cache directory
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		println("defaulting cache dir to temp")
		// Attempt to use the tmp dir
		cacheDir = os.TempDir()
	}

	// Return the environment
	return map[string]string{
		"SIGOROOT":  getenv("SIGOROOT", sigoRoot),
		"SIGOPATH":  getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"SIGOCACHE": getenv("SIGOCACHE", getenv("GOCACHE", filepath.Join(cacheDir, "go-build"))),

		"GOROOT": getenv("GOROOT", goRoot),
		//"GOPATH":   getenv("SIGOPATH", getenv("GOPATH", cwd)),
		"GOCACHE":  getenv("SIGOCACHE", getenv("GOCACHE", filepath.Join(cacheDir, "go-build"))),
		"GOTMPDIR": getenv("SIGOTMPDIR", filepath.Join(cacheDir, "sigo")),
		//"PATH":     os.Getenv("PATH"),
		"CC":      getenv("CC", ""),
		"LD":      getenv("LD", ""),
		"OBJCOPY": getenv("OBJCOPY", ""),
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
