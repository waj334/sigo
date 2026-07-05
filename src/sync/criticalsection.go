package sync

//sigo:extern runtime.gosched
func gosched()

type CriticalSection struct {
	mutex   *Mutex
	state   uint32
	entered bool
}

func NewCriticalSection(mutex *Mutex) CriticalSection {
	return CriticalSection{
		mutex: mutex,
	}
}

func (c *CriticalSection) Begin() {
	if !c.entered {
		if c.mutex != nil {
			for {
				c.state = disableInterrupts()
				if !c.mutex.TryLock() {
					enableInterrupts(c.state)
					gosched()
					continue
				}
				break
			}
		} else {
			// A nil mutex means the critical section is the interrupt mask
			// itself. The previous state is saved so End restores it exactly:
			// a caller that already has interrupts disabled (e.g. the runtime
			// sleep path arming a timer via addsleep) must remain disabled
			// after End, not be forcibly re-enabled inside its own critical
			// section.
			c.state = disableInterrupts()
		}
		c.entered = true
	}
}

func (c *CriticalSection) End() {
	if c.entered {
		c.entered = false
		if c.mutex != nil {
			c.mutex.Unlock()
		}
		enableInterrupts(c.state)
	}
}
