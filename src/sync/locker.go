package sync

type Locker interface {
	Lock()
	Unlock()
}
