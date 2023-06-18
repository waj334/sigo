package builder

var picolibc = library{
	args: []string{
		"-nostdlib",
		"-D_LIBC",
		"-ftls-model=local-exec",
		"-fno-common -fno-stack-protector",
		"-ffunction-sections",
		"-fdata-sections",
		"-Wall",
		"-Wextra",
		"-Werror=implicit-function-declaration",
		"-Werror=vla",
		"-Warray-bounds",
		"-Wold-style-definition",
	},
	filenames: []string{},
}
