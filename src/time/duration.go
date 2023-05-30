package time

const (
	Nanosecond  Duration = 1
	Microsecond          = 1000 * Nanosecond
	Millisecond          = 1000 * Microsecond
	Second               = 1000 * Millisecond
	Minute               = 60 * Second
	Hour                 = 60 * Minute
)

type Duration int64

func Since(t Time) Duration {
	return Duration(int64(Now().t) - int64(t.t))
}

func Until(t Time) Duration {
	return Duration(int64(t.t) - int64(Now().t))
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
