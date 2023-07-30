package SAM

import (
	"fmt"
	"go/format"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func WritePreamble(w io.Writer, deviceName string, pkg string) {
	// TODO: Write the license text

	// Write build tag
	fmt.Fprintf(w, "//go:build %s\n\n", strings.ToLower(deviceName))

	// Write the package
	fmt.Fprintln(w, "package ", pkg)
}

func GenerateInit(deviceName string, out string) (err error) {
	var w strings.Builder
	WritePreamble(&w, deviceName, filepath.Base(out))

	// Write required imports
	fmt.Fprintln(&w, "import (")
	fmt.Fprintln(&w, `_ "runtime/arm/cortexm"`)
	fmt.Fprintln(&w, ")")

	// Format the final output
	var buf []byte
	fname := filepath.Join(out, "init.go")
	src := w.String()
	if buf, err = format.Source([]byte(src)); err != nil {
		return fmt.Errorf("error formatting %s: %v", fname, err)
	}

	// Write the contents to the file
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	f.Write(buf)
	f.Close()

	return nil
}
