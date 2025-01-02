//go:build atsamx5x

package main

import (
	"runtime/arm/cortexm/sam/atsamx5x"
)

func init() {
	atsamx5x.DefaultClocks()
}

func main() {
	//println("Hello, World!")
}
