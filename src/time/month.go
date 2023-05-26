package time

type Month int

var (
	longMonthNames = [12]string{
		"january",
		"february",
		"march",
		"april",
		"may",
		"june",
		"july",
		"august",
		"september",
		"october",
		"november",
		"december",
	}
)

const (
	January Month = 1 + iota
	February
	March
	April
	May
	June
	July
	August
	September
	October
	November
	December
)

func (m Month) String() string {
	if January <= m && m <= December {
		return longMonthNames[m-1]
	}
	return "invalid"
}
