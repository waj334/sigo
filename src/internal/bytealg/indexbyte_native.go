// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm || riscv64

package bytealg

//go:noescape
func IndexByte(b []byte, c byte) int
