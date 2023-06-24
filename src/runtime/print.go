package runtime

import "os"

func _println(args ...any) {
	if os.Stdout != nil {
		// TODO: Convert args to a byte slice
	}
}

func _print(args ...any) {
	if os.Stdout != nil {
		// TODO: Convert args to a byte slice
	}
}
