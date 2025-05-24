#ifndef _ARM_CORTEXM_VARIANT_TD
#define _ARM_CORTEXM_VARIANT_TD

include "base.td"
include "arm/cortexm/interrupts.td"

defvar CortexMTags = [ "arm", "cortexm" ];

class CortexMVariant<string Name, list<MemoryRange> Memories = [], list<Interrupt> Interrupts = [],
    MemoryUnit StackSize = KB<4>, list<string> Tags, string Description = "">
        : Variant<Name, Memories, !listconcat(CortexMInterrupts, Interrupts), StackSize, !listconcat(CortexMTags, Tags),
            Description>;

#endif // _ARM_CORTEXM_VARIANT_TD
