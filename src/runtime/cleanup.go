package runtime

// Cleanup is a handle to a cleanup call.
// It can be used to stop the cleanup call from running.
type Cleanup struct{}

// Stop cancels the cleanup call. Stop will have no effect if the cleanup call
// has already been queued for execution (because ptr became unreachable).
func (c Cleanup) Stop() {}

// AddCleanup attaches a cleanup function to ptr. Some time after ptr is no
// longer reachable, the runtime will call cleanup(arg) in a separate goroutine.
//
// This is a stub implementation for embedded targets — cleanups are not
// executed, but the API is present so that packages like unique compile.
func AddCleanup[T, S any](ptr *T, cleanup func(S), arg S) Cleanup {
	return Cleanup{}
}

// KeepAlive marks its argument as reachable at the point of the call.
// This ensures that the object is not freed, and its finalizer/cleanup is not run,
// before the point in the program where KeepAlive is called.
func KeepAlive(x any) {}
