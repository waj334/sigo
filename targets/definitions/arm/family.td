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

class CortexMArchitectue<string Name, string Arch, string Triple, list<string> Tags, FPU Fpu = NoFPU>
        : ARMArchitecture<Name, Arch, Triple, Fpu> {
    let tags = !listconcat([ "arm", "cortexm" ], Tags);
    let runtimePackages = [ "pkg.si-go.dev/chip/arm/cortexm/runtime" ];
}

def CortexM0 : CortexMArchitectue<"cortexm-m0", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM0Plus : CortexMArchitectue<"cortexm-m0plus", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM1 : CortexMArchitectue<"cortexm-m1", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM23 : CortexMArchitectue<"cortex-m23", "arm", "armv8m.base-none-eabihf", [ "armv8m.base" ]>;
def CortexM3 : CortexMArchitectue<"cortex-m3", "thumb", "armv7m-none-eabi", [ "armv7m", "thumb" ]>;
def CortexM33 : CortexMArchitectue<"cortex-m33", "thumb", "armv8m.main-none-eabihf", [ "armv8m.main", "thumb" ], FPV5_SP_D16>;
def CortexM35P : CortexMArchitectue<"cortex-m35p", "thumb", "armv8m.main-none-eabihf", [ "armv8m.main", "thumb" ], FPV5_SP_D16>;
def CortexM4 : CortexMArchitectue<"cortex-m4", "thumb", "armv7em-none-eabihf", [ "armv7m", "armv7em", "thumb" ], FPV4_SP_D16>;
def CortexM52 : CortexMArchitectue<"cortex-m52", "thumb", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;
def CortexM55 : CortexMArchitectue<"cortex-m55", "thumb", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;
def CortexM7 : CortexMArchitectue<"cortex-m7", "thumb", "armv7em-none-eabihf", [ "armv7m", "armv7em", "thumb" ], FPV5_D16>;
def CortexM85 : CortexMArchitectue<"cortex-m85", "thumb", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;

#endif // _ARM_FAMILY_TD
