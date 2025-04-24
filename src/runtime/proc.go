package runtime

func Gosched() {
	schedulerPause()
}
