package main

import (
	"github.com/spf13/cobra"
	"io/ioutil"
	"log"
	"omibyte.io/sigo/builder"
	"os"
	"path/filepath"
	"strings"
)

const (
	SKIP     = 0
	CONCRETE = 1
	SYMLINK  = 2
)

var (
	sigoRootInclude = map[string]bool{
		"src": true,
	}

	goRootInclude = map[string]bool{
		"src/internal": true,
		"src/builtin":  true,
		"src/unicode":  true,
		"pkg":          true,
	}

	inclusionFileList = []string{
		"bin/go",
		"bin/gofmt",

		"bin/go.exe",
		"bin/gofmt.exe",
	}

	sigoRoot string
	goRoot   string
	rootDir  string

	rootCmd = &cobra.Command{
		Use:   "root",
		Short: "Create a root runtime directory",
		Long:  "Create a directory containing the SiGo runtime. This is for language support in IDEs.",
		Run: func(cmd *cobra.Command, args []string) {
			println("SIGOROOT: ", sigoRoot)
			println("GOROOT: ", goRoot)

			dirMap := map[string]int{}

			// Remove the root
			os.RemoveAll(filepath.ToSlash(rootDir))

			filepath.Walk(sigoRoot, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					log.Fatalf("Failed to access path %s: %v", path, err)
				}

				if info.IsDir() {
					relPath, err := filepath.Rel(sigoRoot, path)
					if err != nil {
						log.Fatalf("Failed to get relative path: %v", err)
					}

					if strings.Contains(relPath, ".git") || strings.Contains(relPath, "CMakeFiles") || strings.Contains(path, "sigo/build") {
						// Skip git folder
						return nil
					}

					includedInSigo := isInclusionList(filepath.ToSlash(relPath), sigoRootInclude)
					includedInGo := isInclusionList(filepath.ToSlash(relPath), goRootInclude)

					// If the directory is in the inclusion list of both directories and exists in both, create it
					if includedInSigo && includedInGo {
						dirMap[path] = CONCRETE
					} else if includedInSigo || includedInGo {
						dirMap[path] = SYMLINK
					}
				}
				return nil
			})

			filepath.Walk(goRoot, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					log.Fatalf("Failed to access path %s: %v", path, err)
				}

				if info.IsDir() {
					relPath, err := filepath.Rel(goRoot, path)
					if err != nil {
						log.Fatalf("Failed to get relative path: %v", err)
					}
					includedInGo := isInclusionList(filepath.ToSlash(relPath), goRootInclude)

					if includedInGo {
						subdirs := strings.Split(filepath.ToSlash(relPath), "/")
						parentPath := ""
						for _, subdir := range subdirs {
							parentPath = filepath.ToSlash(filepath.Join(parentPath, subdir))
							if isInclusionList(parentPath, sigoRootInclude) {
								// Does SiGO have this same directory?
								if _, statErr := os.Stat(filepath.Join(sigoRoot, parentPath)); statErr == nil {
									dirMap[path] = CONCRETE
								} else if isInclusionList(parentPath, goRootInclude) {
									dirMap[path] = SYMLINK
								}
							} else if isInclusionList(parentPath, goRootInclude) {
								dirMap[path] = SYMLINK
							}
						}
					}
				}
				return nil
			})

			// create a copy of the map for iterating
			tempDirMap := make(map[string]int)
			for k, v := range dirMap {
				tempDirMap[k] = v
			}

			/*for path, dirType := range tempDirMap {
				if dirType == CONCRETE {
					// remove all directories that are nested within this concrete directory (children, grandchildren, etc.)
					for nestedPath := range tempDirMap {
						if nestedPath != path && strings.HasPrefix(nestedPath, path+"/") {
							delete(tempDirMap, nestedPath)
						}
					}
				}
			}*/

			// keep only those directories that are immediate children of concrete directories or are not nested within any concrete directory
			reducedDirMap := make(map[string]int)
			for path, dirType := range tempDirMap {
				if dirType == CONCRETE {
					reducedDirMap[path] = dirType
				} else { // dirType == SYMLINK
					parentDir := filepath.Dir(path)
					if parentType, parentExists := tempDirMap[parentDir]; parentExists && parentType == CONCRETE {
						// the parent directory is a concrete directory
						reducedDirMap[path] = dirType
					} else if !parentExists {
						// the parent directory is not in the map, so it's not a concrete directory
						reducedDirMap[path] = dirType
					}
				}
			}

			relToAbs := map[string]string{}
			for path, _ := range reducedDirMap {
				var relPath string
				var err error
				isSigo := false
				if strings.HasPrefix(filepath.ToSlash(path), filepath.ToSlash(sigoRoot)) {
					relPath, err = filepath.Rel(sigoRoot, path)
					isSigo = true
				} else {
					relPath, err = filepath.Rel(goRoot, path)
				}

				if err != nil {
					log.Fatalf("Failed to get relative path: %v", err)
				}

				if _, ok := relToAbs[relPath]; !ok || isSigo {
					relToAbs[filepath.ToSlash(relPath)] = path
				}
			}

			for path, _ := range reducedDirMap {
				var relPath string
				var err error
				if strings.HasPrefix(filepath.ToSlash(path), filepath.ToSlash(sigoRoot)) {
					relPath, err = filepath.Rel(sigoRoot, path)
				} else {
					relPath, err = filepath.Rel(goRoot, path)
				}

				if err != nil {
					log.Fatalf("Failed to get relative path: %v", err)
				}

				subdirs := strings.Split(filepath.ToSlash(relPath), "/")
				parentPath := ""
				for _, subdir := range subdirs {
					parentPath = filepath.ToSlash(filepath.Join(parentPath, subdir))
					targetPath := filepath.Join(rootDir, parentPath)
					mode := reducedDirMap[relToAbs[parentPath]]
					if mode == CONCRETE {
						if err := os.MkdirAll(targetPath, os.ModePerm); err != nil {
							log.Fatalf("Failed to create directory %s: %v", targetPath, err)
						}
					} else if mode == SYMLINK {
						createSymlink(relToAbs[parentPath], targetPath)
					}
				}
			}

			// Symlink files
			symlinkFiles(goRoot, rootDir, inclusionFileList)
		},
	}
)

func init() {
	env := builder.Environment()
	rootCmd.Flags().StringVarP(&sigoRoot, "sigoRoot", "s", env["SIGOROOT"], "SiGo root directory to merge. Default: $SIGOROOT")
	rootCmd.Flags().StringVarP(&goRoot, "goRoot", "g", env["GOROOT"], "Go root directory to merge. Default: $GOROOT")
	rootCmd.Flags().StringVarP(&rootDir, "dir", "o", env["SIGOROOT"]+"/root", "Resulting directory. Default: $SIGOROOT/root")
}

func createDirsForConflicts(srcDir1 string, srcDir2 string, targetDir string, inclusionDirList1 map[string]bool, inclusionDirList2 map[string]bool) {
	filepath.Walk(srcDir1, func(path1 string, info1 os.FileInfo, err1 error) error {
		if err1 != nil {
			log.Fatalf("Failed to access path %s: %v", path1, err1)
		}

		if info1.IsDir() {
			relPath, err := filepath.Rel(srcDir1, path1)
			if err != nil {
				log.Fatalf("Failed to get relative path: %v", err)
			}

			path2 := filepath.Join(srcDir2, relPath)

			// Check if directory exists in srcDir2
			_, err2 := os.Stat(path2)

			// If the directory is in the inclusion list of both directories and exists in both, create it
			if isInclusionList(relPath, inclusionDirList1) && isInclusionList(relPath, inclusionDirList2) && err2 == nil {
				targetPath := filepath.Join(targetDir, relPath)
				if err := os.MkdirAll(targetPath, os.ModePerm); err != nil {
					log.Fatalf("Failed to create directory %s: %v", targetPath, err)
				}
			}
		}

		return nil
	})
}

func createSymlinks(srcDir1, srcDir2, targetDir string, inclusionDirList1, inclusionDirList2 map[string]bool) {
	// First, symlink all directories from the first source directory.
	symlinkDirs(srcDir1, targetDir, inclusionDirList1, isInclusionList)

	// Then, symlink directories from the second source directory.
	symlinkDirs(srcDir2, targetDir, inclusionDirList2, isExplicitlyIncluded)
}

func symlinkDirs(srcDir, targetDir string, inclusionDirList map[string]bool, inclusionCheck func(string, map[string]bool) bool) {
	// Open the source directory.
	dir, err := ioutil.ReadDir(srcDir)
	if err != nil {
		log.Fatalf("Failed to open directory: %v", err)
	}

	// Iterate over each item in the source directory.
	for _, info := range dir {
		// Get the relative path of this directory.
		rel, err := filepath.Rel(srcDir, filepath.Join(srcDir, info.Name()))
		if err != nil {
			log.Fatalf("Failed to get relative path: %v", err)
		}

		// Symlink this directory if it's a directory and either no inclusion list is provided or this directory is in the inclusion list.
		if info.IsDir() && (inclusionDirList == nil || inclusionCheck(filepath.ToSlash(rel), inclusionDirList)) {
			err = createSymlink(filepath.Join(srcDir, info.Name()), filepath.Join(targetDir, info.Name()))
			if err != nil {
				log.Fatalf("Failed to create symlink: %v", err)
			}
		}
	}
}

func symlinkFiles(srcDir, targetDir string, fileList []string) {
	for _, file := range fileList {
		os.MkdirAll(filepath.Dir(filepath.Join(targetDir, file)), os.ModePerm)
		err := createSymlink(filepath.Join(srcDir, file), filepath.Join(targetDir, file))
		if err != nil {
			if os.IsNotExist(err) {
				log.Printf("Skipping non-existent file: %s", file)
				continue
			}
			log.Fatalf("Failed to create symlink for file: %v", err)
		}
	}
}

func createSymlink(src, target string) error {
	// Check if the target exists
	_, err := os.Stat(target)
	if err == nil {
		// Skip if the target already exists
		return nil
	}

	// Create a new symlink.
	return os.Symlink(src, target)
}

func isInclusionList(dirPath string, inclusionList map[string]bool) bool {
	for includedPath := range inclusionList {
		// Check if dirPath is exactly includedPath
		if dirPath == includedPath {
			return true
		}
		// Check if includedPath is a subdirectory of dirPath
		if strings.HasPrefix(includedPath, dirPath+"/") {
			return true
		}
		// Check if dirPath is a subdirectory of includedPath
		if strings.HasPrefix(dirPath+"/", includedPath+"/") {
			return true
		}
	}
	return false
}

func isExplicitlyIncluded(dirPath string, inclusionList map[string]bool) bool {
	// Check if the given directory is in the inclusion list.
	if inclusionList[dirPath] {
		return true
	}

	return false
}
