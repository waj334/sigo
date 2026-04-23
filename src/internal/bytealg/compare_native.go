// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm || riscv64

package bytealg

import _ "unsafe" // For go:linkname

//go:noescape
func Compare(a, b []byte) int

func CompareString(a, b string) int {
	return abigen_runtime_cmpstring(a, b)
}

// The declaration below generates ABI wrappers for functions
// implemented in assembly in this package but declared in another
// package.

//go:linkname abigen_runtime_cmpstring runtime.cmpstring
func abigen_runtime_cmpstring(a, b string) int
