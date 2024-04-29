package ssa

import (
	"encoding/binary"
	"fmt"
	"go/token"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"strings"

	"omibyte.io/sigo/mlir"
)

func (b *Builder) location(p token.Pos) mlir.Location {
	// Get the file positioning information associated with the location.
	pos := b.config.Fset.Position(p)
	if !pos.IsValid() {
		return mlir.LocationFileLineColGet(b.config.Ctx, "<unknown>", 0, 0)
	}

	// Evaluate symlinks.
	filename, err := filepath.EvalSymlinks(pos.Filename)
	if err != nil {
		filename = pos.Filename
	}

	return mlir.LocationFileLineColGet(b.config.Ctx, filename, uint(pos.Line), uint(pos.Column))
}

func (b *Builder) locationHashString(p token.Pos) string {
	pos := b.config.Fset.Position(p)
	if !pos.IsValid() {
		panic("attempting to hash invalid position")
	}

	// Evaluate symlinks.
	filename, err := filepath.EvalSymlinks(pos.Filename)
	if err != nil {
		filename = pos.Filename
	}

	// Create the hasher that will generate the unique hash for the token position.
	hasher := fnv.New32()
	io.WriteString(hasher, filename)
	binary.Write(hasher, binary.LittleEndian, uint32(pos.Line))
	binary.Write(hasher, binary.LittleEndian, uint32(pos.Column))
	return fmt.Sprintf("%X", hasher.Sum32())
}

func (b *Builder) locationString(pos token.Pos) string {
	if pos.IsValid() {
		pos := b.config.Fset.Position(pos)
		if buf, err := os.ReadFile(pos.Filename); err == nil {
			line := strings.Split(string(buf), "\n")[pos.Line-1]
			column := []rune(strings.Repeat(" ", len(line)))
			for i, r := range line {
				switch r {
				case '\t':
					// Copy tabs.
					column[i] = r
				}
				if i == pos.Column-1 {
					column[i] = '^'
				}
			}
			return fmt.Sprintf("%s\n%s", line, string(column))
		}
	}
	return ""
}
