package builder

import (
	"fmt"
	"os"
	"os/exec"
)

type preprocessParameters struct {
	toolchain    Toolchain
	triplet      string
	includePaths []string
	defines      map[string]string
	input        string
	output       string
}

func preprocess(parameters preprocessParameters) error {
	args := []string{
		"-E", "-P",
		"-x", "c",
		"-nostdinc",
		"--target=" + parameters.triplet,
	}

	for _, path := range parameters.includePaths {
		args = append(args, "-I"+path)
	}

	for key, value := range parameters.defines {
		args = append(args, "-D"+key+"="+value)
	}

	args = append(args, parameters.input, "-o", parameters.output)

	cmd := exec.Command(parameters.toolchain.CC, args...)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("preprocessing failed: %w", err)
	}
	return nil
}
