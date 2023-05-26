package time

type Duration int64

func Since(t Time) Duration {
	return 0
}

func Until(t Time) Duration {
	return 0
}

func (d Duration) Abs() Duration {
	return 0
}

func (d Duration) Hours() float64 {
	return 0
}

func (d Duration) Microseconds() int64 {
	return 0
}

func (d Duration) Milliseconds() int64 {
	return 0
}

func (d Duration) Minutes() float64 {
	return 0
}

func (d Duration) Nanoseconds() int64 {
	return 0
}

func (d Duration) Round(m Duration) Duration {
	return 0
}

func (d Duration) Seconds() float64 {
	return 0
}

func (d Duration) String() string {
	return ""
}

func (d Duration) Truncate(m Duration) Duration {
	return 0
}
