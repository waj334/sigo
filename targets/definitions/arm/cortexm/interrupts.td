include "base.td"

defvar CortexMInterrupts = [
    Interrupt<"Reset",          -15,    "Reset Vector, invoked on Power up and warm reset.">,
    Interrupt<"NMI",            -14,    "Non maskable Interrupt, cannot be stopped or preempted.">,
    Interrupt<"Hardfault",      -13,    "Hard Fault, all classes of Fault.">,
    Interrupt<"MemoryManage",   -12,    "Memory Management, MPU mismatch, including Access Violation and No Match.">,
    Interrupt<"BusFault",       -11,    "Bus Fault, Pre-Fetch-, Memory Access Fault, other address/memory related Fault.">,
    Interrupt<"UsageFault",     -10,    "Usage Fault, i.e. Undef Instruction, Illegal State Transition">,
    Interrupt<"SVCall",         -5,     "System Service Call via SVC instruction.">,
    Interrupt<"DebugMonitor",   -4,     "Debug Monitor.">,
    Interrupt<"PendSVHandler",  -2,     "Pendable request for system service.">,
    Interrupt<"SysTickHandler", -1,     "System Tick Timer.">,
];