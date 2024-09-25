package sync

type Pool struct {
	New func() any
}

func (p *Pool) Get() any {
	panic("unimplemented")
}

func (p *Pool) Put(x any) {
	panic("unimplemented")
}
