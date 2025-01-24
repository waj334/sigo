package cortexm4

import (
	"runtime/arm/cortexm"
)

func init() {
	cortexm.SysTickFrequency = 168_000_000
	cortexm.IrqPriorityMask = 0b111
}
