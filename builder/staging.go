package builder

import (
	"errors"
	"gopkg.in/yaml.v3"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
)

type stlFilter struct {
	Include []string `yaml:"include"`
	Exclude []string `yaml:"exclude"`
}

func stageGoRoot(stageDir string, env Env) error {
	goRootPath := env.Value("GOROOT")
	sigoRootPath := env.Value("SIGOROOT")

	// Symlink the "pkg" directory from GOROOT into the staging directory
	goPkgPath := path.Join(goRootPath, "pkg")
	stagedPkgPath := path.Join(stageDir, "pkg")
	err := os.Symlink(goPkgPath, stagedPkgPath)
	if err != nil {
		return err
	}

	// Create the directory for the standard library.
	sigoStlPath := path.Join(sigoRootPath, "src")
	err = os.MkdirAll(sigoStlPath, os.ModeDir)
	if err != nil {
		return err
	}

	// Copy the SiGo standard library into the directory just created.
	filepath.WalkDir(sigoStlPath, func(path string, d fs.DirEntry, err error) error {
		relPath, _ := filepath.Rel(sigoStlPath, path)
		stagedPath := filepath.Join(stageDir, "src", relPath)
		if d.IsDir() {
			if relPath == "." {
				return nil
			}

			// Create this directory.
			err = os.MkdirAll(stagedPath, os.ModeDir)
			if err != nil {
				return err
			}
		} else {
			// Symlink this file into the staged directory.
			err = os.Symlink(path, stagedPath)
			if err != nil {
				return err
			}
		}
		return nil
	})

	// Read the index.yaml file present in the SiGo root directory.
	var b []byte
	b, err = os.ReadFile(path.Join(sigoRootPath, "src", "index.yaml"))
	if err != nil {
		return err
	}

	// Parse the index yaml.
	var filter stlFilter
	err = yaml.Unmarshal(b, &filter)
	if err != nil {
		return err
	}

	// Copy subset of Go standard library into the staging directory.
	goStlPath := path.Join(goRootPath, "src")
	filepath.WalkDir(goStlPath, func(path string, d fs.DirEntry, err error) error {
		relPath, _ := filepath.Rel(goStlPath, path)
		stagedPath := filepath.Join(stageDir, "src", relPath)
		if d.IsDir() {
			if relPath == "." {
				return nil
			}

			// Is this path included?
			if slices.Contains(filter.Include, filepath.ToSlash(relPath)) {
				// Check if the directory does not already exist in the staging directory.
				if _, err := os.Stat(stagedPath); errors.Is(err, os.ErrNotExist) {
					// Create this directory.
					err = os.MkdirAll(stagedPath, os.ModeDir)
					if err != nil {
						return err
					}
				}
			}
		} else {
			if slices.Contains(filter.Include, filepath.ToSlash(filepath.Dir(relPath))) {
				// Check if a file with the same name does not already exist in this directory.
				if _, err := os.Stat(stagedPath); errors.Is(err, os.ErrNotExist) {
					// Symlink this file into the staged directory.
					err = os.Symlink(path, stagedPath)
					if err != nil {
						return err
					}
				}
			}
		}
		return nil
	})

	/*
		// Symlink the standard library from SiGo into the staging directory
		stagedStlPath := path.Join(stageDir, "src")
		if err := os.Symlink(sigoStlPath, stagedStlPath); err != nil {
			return err
		}
	*/

	return nil
}
