package sync

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
			c.mutex.Lock()
		}
		c.state = disableInterrupts()
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
