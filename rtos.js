// ─────────────────────────────────────────────────────────
// Ozone RTOS script: proper register context per goroutine
// Cortex-M (M4/M7)
// ─────────────────────────────────────────────────────────

function getOSName() {
    return "SiGo";
}

function status(value) {
    switch (value) {
        case 0:
            return "Not started";
        case 1:
            return "Ready";
        case 2:
            return "Running";
        case 3:
            return "Panicking";
        case 4:
            return "Recovered";
        case 5:
            return "Parked";
        default:
            return "unknown";
    }
}

function init() {
    Threads.newqueue("Goroutines");
    Threads.setColumns2("Goroutines", "Name", "Stack Usage", "Status");
    Threads.setColor("Goroutines", "Running", "Ready", "Parked");

    Threads.newqueue("Memory");
    Threads.setColumns2("Memory", "", "");
}

function update() {
    Threads.clear();

    if (Threads.shown("Goroutines")) {
        var head = Debug.evaluate("headGoroutine");
        var stackSize = Debug.evaluate("_goroutineStackSize");
        var stackSizeKB = stackSize / 1024;
        if (head != undefined && head != 0) {
            var curr = head;
            while (true) {
                var g = Debug.evaluate("*(runtime_goroutine*)" + curr);
                Threads.add(
                    "goroutine (" + getname(curr) + ")",
                    computeStackUsage(g, stackSize) + "KB / " + stackSizeKB + "KB",
                    status(g.state),
                    curr
                );
                curr = g.next;
                if (curr == head) {
                    break;
                }
            }
        }
    }

    if (Threads.shown("Memory")) {
        var sbrk_start = Debug.evaluate("__malloc_sbrk_top");
        var sbrk_top = Debug.evaluate("__malloc_sbrk_start");
        var heap_start = Debug.evaluate("__heap_start");
        var heap_end = Debug.evaluate("__heap_end");
        var free = Debug.evaluate("__malloc_free_list");

        var total = 0;
        var used = 0;

        if (sbrk_start != undefined && sbrk_top != undefined && free != undefined && heap_start != undefined && heap_end != undefined) {
            used = Debug.evaluate("__malloc_sbrk_top - __malloc_sbrk_start");
            total = Debug.evaluate("&__heap_end - &__heap_start");

            while (free != 0) {
                var pf = Debug.evaluate("*(malloc_chunk*)" + free);
                used -= pf.size;
                free = pf.next;
            }
        }

        total = total / 1024;
        used = used / 1024;

        Threads.add2("Memory", "Heap Usage", used + "KB / " + total + "KB");
    }
}

// ─────────────────────────────────────────────────────────
// Reconstruct registers from a saved goroutine context.
//
// Stack layout from the saved SP upward (PendSvHandler):
//
//   ┌─ saved_SP
//   │  r4, r5, r6, r7, r8, r9, r10, r11   (8 words)
//   │  EXC_RETURN                          (1 word)
//   ├─ +36  (if FPU was active: EXC_RETURN bit4 == 0)
//   │  s16 .. s31                          (16 words, SW FP)
//   ├─ +100  ── PSP at exception entry ──
//   │  r0, r1, r2, r3, r12, lr, pc, xPSR  (8 words, HW basic)
//   ├─ +132  (if FPU was active)
//   │  s0 .. s15, FPSCR, reserved          (18 words, HW FP)
//   ├─ +204
//   │  [4‑byte pad if xPSR bit9 set]
//   └─ original thread SP
//
// When FPU is NOT active:
//   saved_SP +36 → HW basic frame
//   final SP = saved_SP + 68  (+ possible 4‑byte pad)
// ─────────────────────────────────────────────────────────
function decodeSavedRegs(sp) {
    var WORD = 4;

    // ── SW core block: r4‑r11 + EXC_RETURN (9 words) ──
    var p = 0;
    var r4  = TargetInterface.peekWord(sp + p); p += WORD;
    var r5  = TargetInterface.peekWord(sp + p); p += WORD;
    var r6  = TargetInterface.peekWord(sp + p); p += WORD;
    var r7  = TargetInterface.peekWord(sp + p); p += WORD;
    var r8  = TargetInterface.peekWord(sp + p); p += WORD;
    var r9  = TargetInterface.peekWord(sp + p); p += WORD;
    var r10 = TargetInterface.peekWord(sp + p); p += WORD;
    var r11 = TargetInterface.peekWord(sp + p); p += WORD;
    var exc_return = TargetInterface.peekWord(sp + p); p += WORD;
    // p == 36

    // Was the FPU context stacked?  (EXC_RETURN bit4 == 0 → yes)
    var fpActive = ((exc_return & 0x10) == 0);

    // ── Skip SW FP block (s16‑s31) if present ──
    if (fpActive) {
        p += 16 * WORD;   // 64 bytes.  p == 100
    }

    // ── HW basic exception frame: r0 r1 r2 r3 r12 lr pc xPSR ──
    var r0   = TargetInterface.peekWord(sp + p); p += WORD;
    var r1   = TargetInterface.peekWord(sp + p); p += WORD;
    var r2   = TargetInterface.peekWord(sp + p); p += WORD;
    var r3   = TargetInterface.peekWord(sp + p); p += WORD;
    var r12  = TargetInterface.peekWord(sp + p); p += WORD;
    var lr   = TargetInterface.peekWord(sp + p); p += WORD;
    var pc   = TargetInterface.peekWord(sp + p); p += WORD;
    var xpsr = TargetInterface.peekWord(sp + p); p += WORD;

    // ── Skip HW FP frame (s0‑s15, FPSCR, reserved) if present ──
    if (fpActive) {
        p += 18 * WORD;   // 72 bytes
    }

    // ── Account for 8‑byte alignment padding (CCR.STKALIGN) ──
    // If bit 9 of stacked xPSR is set, hardware inserted 4 bytes of
    // padding after the exception frame to maintain 8‑byte alignment.
    if (xpsr & (1 << 9)) {
        p += WORD;
    }

    var final_sp = sp + p;

    var regs = new Array(17);
    regs[0]  = r0;   regs[1]  = r1;   regs[2]  = r2;   regs[3]  = r3;
    regs[4]  = r4;   regs[5]  = r5;   regs[6]  = r6;   regs[7]  = r7;
    regs[8]  = r8;   regs[9]  = r9;   regs[10] = r10;  regs[11] = r11;
    regs[12] = r12;  regs[13] = final_sp;  regs[14] = lr;  regs[15] = pc;
    regs[16] = xpsr;
    return regs;
}

function liveRegs() {
    var names = [
        "R0","R1","R2","R3","R4","R5","R6","R7",
        "R8","R9","R10","R11","R12","SP","LR","PC","xPSR"
    ];
    var regs = new Array(17);
    for (var i = 0; i < names.length; i++) {
        regs[i] = TargetInterface.getRegister(names[i]) >>> 0;
    }
    return regs;
}

function getregs(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    if (g == undefined) {
        return undefined;
    }

    // Check whether this goroutine is the one currently on the CPU.
    // Comparing the goroutine pointer address with currentGoroutine is
    // the most reliable test — the saved SP field is stale for the
    // running goroutine since it was last written during context‑switch‑out.
    var currAddr = Debug.evaluate("currentGoroutine");
    if (currAddr != undefined && currAddr == _g) {
        return liveRegs();
    }

    // Suspended goroutine — reconstruct from saved context frame.
    return decodeSavedRegs(g.stackTop);
}


function getname(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    if (g == undefined) {
        return "<unknown>";
    }
    return Debug.getSymbol(g.__func.f);
}

function computeStackUsage(g, stackSize) {
    if (stackSize == undefined) {
        stackSize = Debug.evaluate("_goroutineStackSize");
    }

    var allocBase = g.stack;
    var currentSP = g.stackTop;

    // For the running goroutine, the actual SP is in the CPU register,
    // not saved in the goroutine struct (which is stale).
    var currAddr = Debug.evaluate("currentGoroutine");
    if (currAddr != undefined) {
        var currg = Debug.evaluate("*(runtime_goroutine*)" + currAddr);
        if (currg != undefined && g.stack == currg.stack) {
            var SP = TargetInterface.getRegister("SP");
            if (SP >= allocBase && SP < (allocBase + stackSize)) {
                currentSP = SP;
            }
        }
    }

    // Stack grows downward: used = (allocBase + stackSize) - SP
    var used = (allocBase + stackSize) - currentSP;
    return (used / 1024);
}

function getMaxStackUsage(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    if (g == undefined) {
        return undefined;
    }
    return computeStackUsage(g);
}