package ssa

import "sync"

type jobQueue struct {
	jobs []*funcData

	done bool
	cond *sync.Cond
	mu   sync.Mutex
}

func newJobQueue(capacity int) *jobQueue {
	q := &jobQueue{jobs: make([]*funcData, 0, capacity)}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (j *jobQueue) push(f *funcData) {
	j.mu.Lock()
	j.jobs = append(j.jobs, f)
	j.mu.Unlock()
	j.cond.Signal()
}

func (j *jobQueue) pushAll(f []*funcData) {
	j.mu.Lock()
	j.jobs = append(j.jobs, f...)
	j.mu.Unlock()
	j.cond.Signal()
}

func (j *jobQueue) pop() *funcData {
	j.mu.Lock()
	defer j.mu.Unlock()
	for len(j.jobs) == 0 && !j.done {
		j.cond.Wait()
	}
	if len(j.jobs) == 0 {
		return nil
	}
	job := j.jobs[0]
	j.jobs = j.jobs[1:]
	return job
}

func (j *jobQueue) close() {
	j.mu.Lock()
	j.done = true
	j.mu.Unlock()
	j.cond.Broadcast()
}
