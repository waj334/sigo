package time

type Time struct {
	t uintptr
}

func Now() Time {
	return Time{
		t: source.Now(),
	}
}

func (t Time) Add(d Duration) Time {
	return Time{}
}

func (t Time) AddDate(years int, months int, days int) Time {
	return Time{}
}

func (t Time) After(u Time) bool {
	return false
}

func (t Time) Clock() (hour, min, sec int) {
	return 0, 0, 0
}

func (t Time) Compare(u Time) int {
	return 0
}

func (t Time) Date() (year int, month Month, day int) {
	return 0, 0, 0
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
