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
    // Init the task table
    Threads.newqueue("Goroutines");
    Threads.setColumns2("Goroutines", "Name", "Stack Usage", "Status");
    Threads.setColor("Goroutines", "Running", "Ready", "Parked");

    // Add memory table.
    Threads.newqueue("Memory");
    Threads.setColumns2("Memory", "", "");
}

function update() {
    Threads.clear();

    if (Threads.shown("Goroutines")) {
        var head = Debug.evaluate("headGoroutine");
        var stackSize = Debug.evaluate("_goroutineStackSize") / 1000;
        if (head != undefined && head != 0) {
            var curr = head;
            while (true) {
                var g = Debug.evaluate("*(runtime_goroutine*)" + curr);
                Threads.add("goroutine (" + getname(curr) + ")", computeStackUsage(g) + "KB / " + stackSize + "KB", status(g.state), curr)
                curr = g.next;
                if (curr == head) {
                    // Stop.
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

        // Calculate heap usage.
        if (sbrk_start != undefined && sbrk_top != undefined && free != undefined && heap_start != undefined && heap_end != undefined) {
            used = Debug.evaluate("__malloc_sbrk_top - __malloc_sbrk_start");
            total = Debug.evaluate("&__heap_end - &__heap_start");

            // Subtract all freed block from the total usage.
            while (free != 0) {
                var pf = Debug.evaluate("*(malloc_chunk*)" + free);
                used -= pf.size;
                free = pf.next;
            }
        }

        // Convert to KB.
        total = total / 1000;
        used = used / 1000;

        Threads.add2("Memory", "Heap Usage", used + "KB / " + total + "KB");
    }
}

function decodeSavedRegs(top) {
    var fpuEnabled = Debug.evaluate("_fpuEnabled");

    var WORD = 4;
    var SZ_SW_CORE= 9 * WORD;   // r4..r11 + EXC_RETURN
    var SZ_SW_FP  = 16 * WORD;  // s16..s31 (software-saved)
    var SZ_HW_FP  = 18 * WORD;  // s0..s15, FPSCR, reserved (hardware)
    var SZ_HW_CORE= 8 * WORD;   // r0 r1 r2 r3 r12 lr pc xPSR

    // Saved layout at g.stackTop:
    //   [r4..r11] (8 words) then HW exception frame
    //   [r0 r1 r2 r3 r12 lr pc xpsr] (8 words)

    var p = 0

    // --- SW core block ---
    var r4          = TargetInterface.peekWord(top + p); p += WORD;
    var r5          = TargetInterface.peekWord(top + p); p += WORD;
    var r6          = TargetInterface.peekWord(top + p); p += WORD;
    var r7          = TargetInterface.peekWord(top + p); p += WORD;
    var r8          = TargetInterface.peekWord(top + p); p += WORD;
    var r9          = TargetInterface.peekWord(top + p); p += WORD;
    var r10         = TargetInterface.peekWord(top + p); p += WORD;
    var r11         = TargetInterface.peekWord(top + p); p += WORD;
    var exc_return  = TargetInterface.peekWord(top + (WORD * 8)); p += WORD;

    // Was FP HW frame present?  (bit4 == 0 means yes)
    var fpHwPresent = ((exc_return & 0x10) == 0);

    // --- Skip over the optional HW FP extended frame ---
    if (fpHwPresent) {
        TargetInterface.message("FPU was active");

        // Skip SW FP block (s16..s31) and HW FP extended frame
        p += SZ_SW_FP + SZ_HW_FP;
    }

    // --- HW core exception frame ---
    var r0   = TargetInterface.peekWord(top + p); p += WORD;
    var r1   = TargetInterface.peekWord(top + p); p += WORD;
    var r2   = TargetInterface.peekWord(top + p); p += WORD;
    var r3   = TargetInterface.peekWord(top + p); p += WORD;
    var r12  = TargetInterface.peekWord(top + p); p += WORD;
    var lr   = TargetInterface.peekWord(top + p); p += WORD;
    var pc   = TargetInterface.peekWord(top + p); p += WORD;
    var xpsr = TargetInterface.peekWord(top + p); p += WORD;

    // SP should be the PSP *after* stacking (what the thread would see on resume).
    var sp = top + p;

    var regs = new Array(17);
    regs[0]  = r0;  regs[1]  = r1;  regs[2]  = r2;  regs[3]  = r3;
    regs[4]  = r4;  regs[5]  = r5;  regs[6]  = r6;  regs[7]  = r7;
    regs[8]  = r8;  regs[9]  = r9;  regs[10] = r10; regs[11] = r11;
    regs[12] = r12; regs[13] = sp;  regs[14] = lr;  regs[15] = pc;
    regs[16] = xpsr;
    return regs;
}

function liveRegs() {
    // Pull the live CPU regs for the currently running goroutine.
    // Order: R0..R15, xPSR
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
    var SP = TargetInterface.getRegister("SP");
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);

    // If this goroutine is actually running, feed Ozone the live CPU regs.
    var currg = Debug.evaluate("*(runtime_goroutine*)currentGoroutine");
    if (currg != undefined && g.stack == currg.stack) {
        return liveRegs();
    }

    // Otherwise, reconstruct from the saved context frame.
    return decodeSavedRegs(g.stackTop);
}


function getname(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    if (g == undefined) {
        return "<unknown>";
    }
    return Debug.getSymbol(g.__func.f);
}

function computeStackUsage(g) {
    var stackSize = Debug.evaluate("_goroutineStackSize");
    var top = g.stackTop;
    var currg = Debug.evaluate("*(runtime_goroutine*)currentGoroutine");
    var SP = TargetInterface.getRegister("SP");
    if (currg != undefined && g.stack == currg.stack && (SP >= currg.stack && SP < currg.stackTop)) {
        // Get the top of the stack from the SP register.
        top = SP;
    }
    return stackSize - (top - g.stack);
}

function getMaxStackUsage(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    if (g == undefined) {
        return undefined;
    }
    return computeStackUsage(g)
}