package pin

type Direction int
type IRQMode int
type PullMode int

type InterruptHandler func(Pin)
