package godebugs

type Info struct {
	Name      string
	Package   string
	Changed   int
	Old       string
	Opaque    bool
	Immutable bool
}

var All []Info

type RemovedInfo struct {
	Name    string
	Removed int
}

var Removed []RemovedInfo

func Lookup(name string) *Info {
	return nil
}
