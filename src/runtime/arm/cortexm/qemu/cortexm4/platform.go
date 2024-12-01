package cortexm4

import (
	"runtime/arm/cortexm"
)

func init() {
	cortexm.SYSTICK_FREQUENCY = 168_000_000
	cortexm.NPRIORITY_BITS = 3
}
