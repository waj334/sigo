.extern runtime.headGoroutine
.extern runtime.currentGoroutine
.extern runtime._goroutineStackSize

.global sigo_headGoroutine
.global sigo_currentGoroutine
.global sigo_goroutineStackSize

.set sigo_headGoroutine,     runtime.headGoroutine
.set sigo_currentGoroutine,  runtime.currentGoroutine
.set sigo_goroutineStackSize, runtime._goroutineStackSize