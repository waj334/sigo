//go:build generic

package rtc

import "runtime"

type RTC interface {
	runtime.TimeSource
}
