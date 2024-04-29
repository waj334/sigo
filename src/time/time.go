package time

import (
	"runtime"
)

var (
	source runtime.TimeSource = runtime.SysTickSource{}
)

func SetSource(src runtime.TimeSource) {
	source = src
}

type Time struct {
	t uint64
}

var (
	yearNs   uint64 = 3.154e+16
	monthNs  uint64 = 2.628e+15
	dayNs    uint64 = 86399905315173
	hourNs   uint64 = 3599996054799
	minuteNs uint64 = 59999934247
	secondNs uint64 = 999998904
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

func (t Time) In(loc *Location) Time {
	return Time{}
}

func (t Time) IsDST() bool {
	return false
}

func (t Time) IsZero() bool {
	return false
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
	return 0
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
	return 0
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
	return 0
}

func (t Time) UnixMicro() int64 {
	return 0
}

func (t Time) UnixMilli() int64 {
	return 0
}

func (t Time) UnixNano() int64 {
	return 0
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
