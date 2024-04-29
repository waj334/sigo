package builder

import (
	"os/exec"
	"path/filepath"
)

type Toolchain struct {
	CC      string
	LD      string
	ObjCopy string
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
