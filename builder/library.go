package builder

import (
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

type library struct {
	args      []string
	filenames []string
}

type libraryConfig struct {
	triple string
	cpu    string
	fpu    string
	float  string
	env    Env
}

func (l library) compile(config libraryConfig, buildDir string) (objects []string, err error) {
	// Locate clang
	executablePostfix := ""
	if runtime.GOOS == "windows" {
		executablePostfix = ".exe"
	}
	clangExecutable := filepath.Join(config.env.Value("SIGOROOT"), "bin/clang"+executablePostfix)
	if _, err = os.Stat(clangExecutable); os.IsNotExist(err) {
		// Try build folder if this is a dev build
		clangExecutable = filepath.Join(config.env.Value("SIGOROOT"), "build/llvm-build/bin/clang"+executablePostfix)
		if _, err = os.Stat(clangExecutable); os.IsNotExist(err) {
			return nil, errors.New("could not locate clang in SIGOROOT")
		}
	}

	if len(config.fpu) == 0 {
		config.fpu = "none"
	}

	if len(config.float) == 0 {
		config.float = "soft"
	}

	// Compile each source file
	for _, fname := range l.filenames {
		dir, file := filepath.Split(fname)

		// Create hash from path and filename
		h := fnv.New32()
		h.Write([]byte(dir))
		h.Write([]byte(file))

		// Format the object name using the hash
		objectFile := fmt.Sprintf("%d.o", h.Sum32())

		// Format the output path for this object file
		out := filepath.Join(buildDir, objectFile)

		// Build command arg list
		args := []string{
			fmt.Sprintf("-target=%s", config.triple),
			fmt.Sprintf("-mcpu=%s", config.cpu),
			fmt.Sprintf("-mfpu=%s", config.fpu),
			fmt.Sprintf("-mfloat=%s", config.float),
			"-o", out,
			"-c", fname,
		}
		args = append(args, l.args...)

		// Compile
		clangCmd := exec.Command(clangExecutable, args...)
		clangCmd.Stdout = os.Stdout
		clangCmd.Stderr = os.Stderr
		if err = clangCmd.Run(); err != nil {
			return nil, errors.Join(ErrClangFailed, err)
		}

		// Append object file to output
		objects = append(objects, out)
	}
	return
}
