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

function getregs(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)" + _g);
    var regs = new Array(16);
    if (g == undefined) {
        return [];
    }

    for (i = 0; i < 16; i++) {
        regs[i] = TargetInterface.peekWord(g.stackTop + i * 4);
    }
    return regs;
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