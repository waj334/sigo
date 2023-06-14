package time

//sigo:extern sleep runtime.sleep
func sleep(d uint64)

func Sleep(d Duration) {
	if d > 0 {
		sleep(uint64(d))
	}
}
