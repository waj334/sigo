package _go

//go:linkname runtime.println
func _println(args []any)

//go:linkname runtime.print
func _print(args []any)
