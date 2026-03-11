package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"pkg.si-go.dev/sigo/builder"
)

var cleanCmd = &cobra.Command{
	Use:   "clean",
	Short: "Cleans build cache",
	Long:  "Cleans build cache",
	Run: func(cmd *cobra.Command, args []string) {
		env, err := builder.Environment()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Toolchain error: %v\n", err)
			return
		}

		cacheDir := env.Value("SIGOCACHE")
		directories, err := filepath.Glob(filepath.Join(cacheDir, "sigo-build*"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Toolchain error: %v\n", err)
			return
		}

		// Remove all build directories.
		for _, dir := range directories {
			err = os.RemoveAll(dir)
			if err != nil {
				fmt.Fprintf(os.Stderr, "An error occurred while removing '%s': %v\n", dir, err)
			}
		}

		// Remove the third-party build cache.
		thirdpartyDir := filepath.Join(cacheDir, "thirdparty")
		err = os.RemoveAll(thirdpartyDir)
		if err != nil && !errors.Is(err, os.ErrExist) {
			fmt.Fprintf(os.Stderr, "An error occurred while removing '%s': %v\n", thirdpartyDir, err)
		}
	},
}
