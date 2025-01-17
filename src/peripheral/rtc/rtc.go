//go:build generic

package rtc

import "time"

type RTC interface {
	time.Source
}
