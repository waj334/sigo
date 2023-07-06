package rand

// rngSource LCG-base random number generator
type rngSource struct {
	state uint64
}

func (r *rngSource) Int63() int64 {
	return int64(r.Uint64() & ^uint64(1<<63))
}

func (r *rngSource) Seed(seed int64) {
	r.state = uint64(seed)
}

func (r *rngSource) Uint64() uint64 {
	// Parameters from MMIX by Donald Knuth
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return r.state
}
