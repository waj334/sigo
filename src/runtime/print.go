package runtime

import "os"

//go:export _println runtime.println
func _println(args ...any) {
	if os.Stdout != nil {
		// TODO: Convert args to a byte slice
	}
}

//go:export _print runtime.print
func _print(args ...any) {
	if os.Stdout != nil {
		// TODO: Convert args to a byte slice
	}
}
