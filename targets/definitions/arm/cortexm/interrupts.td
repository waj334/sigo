#ifndef _ARM_CORTEXM_INTERRUPTS_TD
#define _ARM_CORTEXM_INTERRUPTS_TD

include "base.td"

defvar CortexMInterrupts = [
    Interrupt<"Reset",          -15,    "Reset Vector, invoked on Power up and warm reset.">,
    Interrupt<"NMI",            -14,    "Non maskable Interrupt, cannot be stopped or preempted.">,
    Interrupt<"Hardfault",      -13,    "Hard Fault, all classes of Fault.">,
    Interrupt<"Memory_Manage",   -12,    "Memory Management, MPU mismatch, including Access Violation and No Match.">,
    Interrupt<"Bus_Fault",       -11,    "Bus Fault, Pre-Fetch-, Memory Access Fault, other address/memory related Fault.">,
    Interrupt<"Usage_Fault",     -10,    "Usage Fault, i.e. Undef Instruction, Illegal State Transition">,
    Interrupt<"SV_Call",         -5,     "System Service Call via SVC instruction.">,
    Interrupt<"Debug_Monitor",   -4,     "Debug Monitor.">,
    Interrupt<"Pend_SV",  -2,     "Pendable request for system service.">,
    Interrupt<"Sys_Tick", -1,     "System Tick Timer.">,
];

#endif // _ARM_CORTEXM_INTERRUPTS_TD