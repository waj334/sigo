package common

//sigo:export wake runtime.wake
func wake(t uint64)

func alarm(t uint64) {
	wake(t * TimeScale)
}

//sigo:export nanotime runtime.nanotime
func nanotime() uint64 {
	return TIM2.Tick() * TimeScale
}

//sigo:export addsleep runtime.addsleep
func addsleep(deadline uint64) {
	TIM2.SetAlarm(deadline/TimeScale, alarm)
}
