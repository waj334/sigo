#ifndef _ARM_FAMILY_TD
#define _ARM_FAMILY_TD

include "base.td"

// Define supported ARM family.
def ARM : Family<"arm">;
class ARMArchitecture<string Name, string Arch, string Triple, FPU Fpu = NoFPU, MPU Mpu = NoMPU>
        : Architecture<Name, ARM, Arch, Triple, Fpu, Mpu>;

def CortexM0 : ARMArchitecture<"cortexm-m0", "arm", "armv6m-none-eabi">;
def CortexM0Plus : ARMArchitecture<"cortexm-m0plus", "arm", "armv6m-none-eabi", NoFPU, MPUPresent>;
def CortexM1 : ARMArchitecture<"cortexm-m1", "arm", "armv6m-none-eabi">;
def CortexM23 : ARMArchitecture<"cortex-m23", "arm", "armv8m.base-none-eabihf">;
def CortexM3 : ARMArchitecture<"cortex-m3", "thumb", "armv7m-none-eabi">;
def CortexM33 : ARMArchitecture<"cortex-m33", "thumb", "armv8m.main-none-eabihf", SP>;
def CortexM35P : ARMArchitecture<"cortex-m35p", "thumb", "armv8m.main-none-eabihf", SP>;
def CortexM4 : ARMArchitecture<"cortex-m4", "thumb", "armv7em-none-eabi", SP>;
def CortexM52 : ARMArchitecture<"cortex-m52", "thumb", "armv81m-none-eabi", DP>;
def CortexM55 : ARMArchitecture<"cortex-m55", "thumb", "armv81m-none-eabi", DP>;
def CortexM7 : ARMArchitecture<"cortex-m7", "thumb", "armv7em-none-eabi", DP>;
def CortexM85 : ARMArchitecture<"cortex-m85", "thumb", "armv81m-none-eabi", DP>;

#endif // _ARM_FAMILY_TD
