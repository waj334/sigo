package runtime

//sigo:export wake runtime.wake
//sigo:linkage wake weak
func wake(uint64) {
	// By default, this does nothing when the time package is not used by the program.
}
