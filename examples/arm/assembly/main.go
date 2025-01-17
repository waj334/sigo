package main

import (
	"asm"
	"asm/register"
)

func main() {
	x := 2
	y := 1

	var sum, diff int

	asm.Inline(`
	adds {sum}, {x}, {y}
	sub {diff}, {sum}, {y}`,
		asm.Out(register.R, &sum, asm.Reserve),
		asm.InOut(register.R, &diff, asm.Reserve),
		asm.In(x),
		asm.In(y))

	use(sum)
	use(diff)
}

func use(v any) {
}
