package builder

import (
	"bufio"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type Toolchain struct {
	CC      string
	LD      string
	ObjCopy string
}

func (t *Toolchain) includePaths(env Env) ([]string, error) {
	// TODO: Detect the compiler more intelligently by parsing its version text.
	var args []string
	cc := filepath.Base(t.CC)
	if strings.Contains(cc, "clang") && runtime.GOOS == "windows" {
		args = append(args, "--target=x86_64-pc-windows-gnu")
	}

	cgoArgs := strings.Split(env.Value("CGO_FLAGS"), " ")
	if len(cgoArgs) > 0 {
		args = append(args, cgoArgs...)
	}
	args = append(args, "-v", "-E", "-x", "c", "nul")

	cmd := exec.Command(t.CC, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}

	var paths []string
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	capture := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "#include <...> search starts here:") {
			capture = true
			continue
		}

		if strings.Contains(line, "End of search list.") {
			break
		}

		if capture {
			paths = append(paths, strings.TrimSpace(line))
		}
	}

	return paths, nil
}

func findToolchain(env Env) (Toolchain, error) {
	cc := env.Value("CC")
	if len(cc) == 0 {
		var err error
		if cc, err = findExecutable("clang"); err != nil {
			// Fallback to GCC.
			cc, err = findExecutable("gcc")
		}

		if err != nil {
			return Toolchain{}, err
		}
	}

	ld := env.Value("LD")
	if len(ld) == 0 {
		var err error
		if ld, err = findExecutable("ld.lld"); err != nil {
			// Fallback to LD.
			ld, err = findExecutable("ld")
		}

		if err != nil {
			return Toolchain{}, err
		}
	}

	objcopy := env.Value("OBJCOPY")
	if len(objcopy) == 0 {
		var err error
		if objcopy, err = findExecutable("llvm-objcopy"); err != nil {
			// Fallback to objcopy.
			objcopy, err = findExecutable("objcopy")
		}

		if err != nil {
			return Toolchain{}, err
		}
	}

	return Toolchain{
		CC:      cc,
		LD:      ld,
		ObjCopy: objcopy,
	}, nil
}

func findExecutable(cmd string) (string, error) {
	fname, err := exec.LookPath(cmd)
	if err == nil {
		fname, err = filepath.Abs(fname)
	}
	return fname, err
}
