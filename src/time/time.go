package time

var source TimeSource

func SetSource(src TimeSource) {
	source = src
}

type Time struct {
	t uint64
}

const (
	secondNs = 1000000000
	minuteNs = 60 * secondNs
	hourNs   = 60 * minuteNs
	dayNs    = 24 * hourNs

	// TODO: These cannot be constant.
	monthNs = 2.628e+15
	yearNs  = 3.154e+16

	secondsPerMinute = 60
	secondsPerHour   = 60 * secondsPerMinute
	secondsPerDay    = 24 * secondsPerHour
	secondsPerWeek   = 7 * secondsPerDay
)

func Now() Time {
	return Time{
		t: source.Now(),
	}
}

func (t Time) Add(d Duration) Time {
	newNs := int64(t.t) + d.Nanoseconds()
	if newNs < 0 {
		// Do not allow underflow of uint64
		newNs = 0
	}

	return Time{
		t: uint64(newNs),
	}
}

func (t Time) AddDate(years int, months int, days int) Time {
	return Time{
		t: t.t + (yearNs * uint64(years)) + (monthNs * uint64(months)) + (dayNs * uint64(days)),
	}
}

func (t Time) After(u Time) bool {
	return t.t > u.t
}

func (t Time) Before(u Time) bool {
	return t.t < u.t
}

func (t Time) Clock() (hour, min, sec int) {
	sec = int(t.t/secondNs) % 60
	min = int(t.t/minuteNs) % 60
	hour = int(t.t/hourNs) % 24
	return
}

func (t Time) Compare(u Time) int {
	switch {
	case t.t < u.t:
		return -1
	case t.t > u.t:
		return +1
	}
	return 0
}

func (t Time) Date() (year int, month Month, day int) {
	return
}

func (t Time) Day() int {
	return 0
}

func (t Time) Equal(u Time) bool {
	return false
}

func (t Time) Format(layout string) string {
	return ""
}

func (t Time) GoString() string {
	return ""
}

func (t Time) GobEncode() ([]byte, error) {
	return nil, nil
}

func (t Time) Hour() int {
	return int(t.t/hourNs) % 24
}

func (t Time) In(loc *Location) Time {
	return Time{}
}

func (t Time) IsDST() bool {
	return false
}

func (t Time) IsZero() bool {
	return t.t == 0
}

func (t Time) Local() Time {
	return Time{}
}

func (t Time) Location() *Location {
	return nil
}

func (t Time) MarshalBinary() ([]byte, error) {
	return nil, nil
}

func (t Time) MarshalJSON() ([]byte, error) {
	return nil, nil
}

func (t Time) MarshalText() ([]byte, error) {
	return nil, nil
}

func (t Time) Minute() int {
	return int(t.t/minuteNs) % 60
}

func (t Time) Month() Month {
	return 0
}

func (t Time) Nanosecond() int {
	return 0
}

func (t Time) Round() Time {
	return Time{}
}

func (t Time) Second() int {
	return int(t.t/secondNs) % 60
}

func (t Time) String() string {
	return ""
}

func (t Time) Truncate(d Duration) Time {
	return Time{}
}

func (t Time) UTC() Time {
	return Time{}
}

func (t Time) Unix() int64 {
	return t.UnixMilli() * 1000
}

func (t Time) UnixMicro() int64 {
	return t.UnixNano() * 1000
}

func (t Time) UnixMilli() int64 {
	return t.UnixMicro() * 1000
}

func (t Time) UnixNano() int64 {
	return int64(t.t)
}

func (t *Time) UnmarshalBinary(data []byte) error {
	return nil
}

func (t *Time) UnmarshalJSON(data []byte) error {
	return nil
}

func (t *Time) UnmarshalText(data []byte) error {
	return nil
}
