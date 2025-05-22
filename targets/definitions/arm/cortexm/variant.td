#ifndef _ARM_CORTEXM_VARIANT_TD
#define _ARM_CORTEXM_VARIANT_TD

include "base.td"
include "arm/cortexm/interrupts.td"

class CortexMVariant<string Name, list<MemoryRange> Memories = [], list<Interrupt> Interrupts = [],
    MemoryUnit StackSize = KB<4>, string Description = "">
        : Variant<Name, Memories, !listconcat(CortexMInterrupts, Interrupts), StackSize, Description>;

#endif // _ARM_CORTEXM_VARIANT_TD
