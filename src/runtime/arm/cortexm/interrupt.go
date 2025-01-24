package cortexm

import (
	"runtime/arm/cortexm/support/nvic"
)

var IrqPriorityMask uint8 = 0xFF

type Interrupt int16

func (i Interrupt) EnableIRQ() {
	nvic.Nvic.Iser[i>>5].SetSetena(1 << (i & 0x1F))
}

func (i Interrupt) DisableIRQ() {
	nvic.Nvic.Icer[i>>5].SetClrena(1 << (i & 0x1F))
}

func (i Interrupt) SetPriority(priority uint8) {
	nvic.Nvic.Ip[i].SetIpr(priority & IrqPriorityMask)
}
