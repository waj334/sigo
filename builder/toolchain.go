package builder

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
		if cc, err = findVersionedExecutable("clang", "-"); err != nil {
			// Fallback to GCC.
			cc, err = findVersionedExecutable("gcc", "-")
		}

		if err != nil {
			return Toolchain{}, err
		}
	}

	ld := env.Value("LD")
	if len(ld) == 0 {
		var err error
		if ld, err = findVersionedExecutable("ld.lld", "-"); err != nil {
			// Fallback to LD.
			ld, err = findVersionedExecutable("ld", "-")
		}

		if err != nil {
			return Toolchain{}, err
		}
	}

	objcopy := env.Value("OBJCOPY")
	if len(objcopy) == 0 {
		var err error
		if objcopy, err = findVersionedExecutable("llvm-objcopy", "-"); err != nil {
			// Fallback to objcopy.
			objcopy, err = findVersionedExecutable("objcopy", "-")
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

func findVersionedExecutable(cmd string, sep string) (string, error) {
	pathEnv := os.Getenv("PATH")
	paths := strings.Split(pathEnv, string(os.PathListSeparator))

	var exe string
	for _, pathPrefix := range paths {
		wildcard := filepath.Join(pathPrefix, cmd) + sep + "*"
		if versions, err := filepath.Glob(wildcard); err == nil {
			maxVersion := 0
			for _, version := range versions {
				s := strings.Split(version, cmd+sep)
				if len(s) > 1 {
					if v, err := strconv.Atoi(s[1]); err == nil && v > maxVersion {
						maxVersion = v
						exe = version
					}
				} else {
					// Just use this version directly.
					exe = version
				}
			}
		}
	}

	if len(exe) != 0 {
		return exe, nil
	}

	return "", fmt.Errorf("could not find executable for %s", cmd)
}

func clangTargetFlags(
	triple string,
	cpu string,
	fpu string,
	floatMode string,
) []string {
	args := []string{
		"--target=" + triple,
	}

	if cpu != "" {
		args = append(args, "-mcpu="+cpu)
	}

	// Cortex-M always executes Thumb instructions.
	args = append(args, "-mthumb")

	if fpu != "" && fpu != "nofpu" && floatMode == "hardfp" {
		args = append(
			args,
			"-mfpu="+fpu,
			"-mfloat-abi=hard",
		)
	} else {
		args = append(args, "-mfloat-abi=soft")
	}

	return args
}
