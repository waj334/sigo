# SSA language tests (on-device)

Each subdirectory is a standalone program that exercises a language feature
end-to-end. Programs share `common/` for board bring-up (UART for stdout,
TIM2 for `time.Sleep`) and pass/fail markers.

## Layout

- `common/` — UART/TIM2 setup helpers + `Begin/Pass/Fail/AssertEq/Done` API.
- `<test>/main.go` — one program per test. Calls `common.Setup()` from `init`,
  runs assertions, prints `DONE`, halts.
- `Makefile` — builds every `<test>/main.go` into `bin/<test>.elf`.

## Build

```
make                    # build all tests
make rangefunc_basic    # build a single test
make clean
```

Override defaults via env: `CPU=`, `FLOAT=`, `STACK_SIZE=`, `OPT=`, `DBG=`,
`SIGOC=`. Defaults match the cypress driver example (STM32H747XI, hardfp).

## Run

Flash `bin/<test>.elf` to the device. Output appears on UART1 (PA9 TX,
PB7 RX, 115200 8N1).

Each test prints lines of the form:
```
BEGIN <test_name>
PASS <label>
FAIL <label>: <message>
DONE
```

The `DONE` marker indicates the program finished cleanly. Absence of `DONE`
means the program hung, panicked, or hit a fault before the assertions
completed.

## Current tests

Phase 1 — range-over-func natural completion:

- `rangefunc_basic` — single-value Seq, body accumulates yielded ints.
- `rangefunc_pairs` — two-value Seq2, body accumulates keys and values.
- `rangefunc_zero` — zero-value Seq0, body counts iterations.

Phase 2A — `break` and `continue` inside a range-over-func body:

- `rangefunc_break` — `break` returns false from the yield closure; the
  iterator stops; control falls through past the range statement.
- `rangefunc_continue` — `continue` branches to a closure-local block that
  returns true; the iterator yields the next value.

Not yet supported (later phases):

- `return` inside a range-over-func body (Phase 2B — needs state alloca and
  result-temp captures).
- Labeled break/continue across nested ranges (Phase 3).
- `iter.Pull` / `iter.Pull2` (Phase 6+ — needs runtime coroutine primitives).
