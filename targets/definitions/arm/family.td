#ifndef _ARM_FAMILY_TD
#define _ARM_FAMILY_TD

include "base.td"

// FPU definitions.
def VFP2 : FPU<"vfpv2">;
def VFPV3 : FPU<"vfpv3">;
def VFPV3_D16 : FPU<"vfpv3-d16">;
def VFPV4 : FPU<"vfpv4">;
def VFPV4_D16 : FPU<"vfpv4-d16">;
def FPV4_SP_D16 : FPU<"fpv4-sp-d16">;
def FPV4_DP_D16 : FPU<"vfpv4-d16">;
def FPV5_SP_D16 : FPU<"fpv5-sp-d16">;
def FPV5_D16 : FPU<"fpv5-d16">;
def NEON : FPU<"neon">;

// Define supported ARM family.
def ARM : Family<"arm">;
class ARMArchitecture<string Name, string Arch, string Triple, FPU Fpu = NoFPU>
        : Architecture<Name, ARM, Arch, Triple, Fpu>;

def CortexM0 : ARMArchitecture<"cortexm-m0", "arm", "armv6m-none-eabi">;
def CortexM0Plus : ARMArchitecture<"cortexm-m0plus", "arm", "armv6m-none-eabi">;
def CortexM1 : ARMArchitecture<"cortexm-m1", "arm", "armv6m-none-eabi">;
def CortexM23 : ARMArchitecture<"cortex-m23", "arm", "armv8m.base-none-eabihf">;
def CortexM3 : ARMArchitecture<"cortex-m3", "thumb", "armv7m-none-eabi">;
def CortexM33 : ARMArchitecture<"cortex-m33", "thumb", "armv8m.main-none-eabihf", FPV5_SP_D16>;
def CortexM35P : ARMArchitecture<"cortex-m35p", "thumb", "armv8m.main-none-eabihf", FPV5_SP_D16>;
def CortexM4 : ARMArchitecture<"cortex-m4", "thumb", "armv7em-none-eabihf", FPV4_SP_D16>;
def CortexM52 : ARMArchitecture<"cortex-m52", "thumb", "armv81m.main-none-eabihf", FPV5_D16>;
def CortexM55 : ARMArchitecture<"cortex-m55", "thumb", "armv81m.main-none-eabihf", FPV5_D16>;
def CortexM7 : ARMArchitecture<"cortex-m7", "thumb", "armv7em-none-eabihf", FPV5_D16>;
def CortexM85 : ARMArchitecture<"cortex-m85", "thumb", "armv81m.main-none-eabihf", FPV5_D16>;

#endif // _ARM_FAMILY_TD
