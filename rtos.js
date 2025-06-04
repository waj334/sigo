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
    Threads.newqueue("Tasks");
    Threads.setColumns("Name", "Status");
    Threads.setColor("Status", "Not started", "Ready", "Running", "Panicking", "Recovered", "Parked");
}

function update() {
    Threads.clear();

    var head = Debug.evaluate("headGoroutine");
    if (head != undefined && head != 0) {
        var curr = head;
        var index = 0;
        while (true) {
            var g = Debug.evaluate("*(runtime_goroutine*)"+curr);
            Threads.add("goroutine (" + getname(curr) + ")", status(g.state), curr)
            curr = g.next;
            if (curr == head) {
                // Stop.
                break;
            }
        }
    }
}

function getregs(_g) {
    var g = Debug.evaluate("*(runtime_goroutine*)"+_g);
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
    var g = Debug.evaluate("*(runtime_goroutine*)"+_g);
    if (g == undefined) {
        return "<unknown>";
    }
    return Debug.getSymbol(g.__func.f);
}

function getMaxStackUsage(_g)
{
    var g = Debug.evaluate("*(runtime_goroutine*)"+_g);
    if (g == undefined) {
        return undefined;
    }
    return g.stackTop - g.stack;
}