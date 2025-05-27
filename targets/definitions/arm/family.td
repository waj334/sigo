#ifndef _ARM_FAMILY_TD
#define _ARM_FAMILY_TD

include "base.td"

// FPU definitions.
def VFP2 : FPU<"vfpv2", [ "vfp2" ]>;
def VFPV3 : FPU<"vfpv3", [ "vfp3" ]>;
def VFPV3_D16 : FPU<"vfpv3-d16", [ "vfp3", "vfp3d16" ]>;
def VFPV3_SP_D16 : FPU<"vfpv3-d16", [ "vfp3", "vfp3d16sp" ]>;
def VFPV4 : FPU<"vfpv4", [ "vfp4" ]>;
def VFPV4_D16 : FPU<"vfpv4-d16", [ "vfp4", "vfp4d16" ]>;
def FPV4_SP_D16 : FPU<"fpv4-sp-d16", [ "vfp4", "vfp4d16sp" ]>;
def FPV4_DP_D16 : FPU<"vfpv4-d16", [ "vfp4", "vfp4d16" ]>;
def FPV5 : FPU<"vfpv5", [ "fp-armv8" ]>;
def FPV5_SP_D16 : FPU<"fpv5-sp-d16", [ "fp-armv8", "fp-armv8d16" ]>;
def FPV5_D16 : FPU<"fpv5-d16", [ "fp-armv8", "fp-armv8d16sp" ]>;
def NEON : FPU<"neon", [ "neon" ]>;

// Define supported ARM family.
def ARM : Family<"arm">;
class ARMArchitecture<string Name, string Arch, string Triple, int Alignment, FPU Fpu = NoFPU>
        : Architecture<Name, ARM, Arch, Triple, Alignment, Fpu>;

class CortexMArchitectue<string Name, string Arch, string Triple, list<string> Tags, FPU Fpu = NoFPU>
        : ARMArchitecture<Name, Arch, Triple, 4, Fpu> {
    let tags = !listconcat([ "arm", "cortexm" ], Tags);
    let runtimePackages = [ "pkg.si-go.dev/chip/arm/cortexm/runtime" ];
}

def CortexM0 : CortexMArchitectue<"cortexm-m0", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM0Plus : CortexMArchitectue<"cortexm-m0plus", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM1 : CortexMArchitectue<"cortexm-m1", "arm", "armv6m-none-eabi", [ "armv6m" ]>;
def CortexM23 : CortexMArchitectue<"cortex-m23", "arm", "armv8m.base-none-eabihf", [ "armv8m.base" ]>;
def CortexM3 : CortexMArchitectue<"cortex-m3", "thumb2", "armv7m-none-eabi", [ "armv7m", "thumb" ]>;
def CortexM33 : CortexMArchitectue<"cortex-m33", "thumb2", "armv8m.main-none-eabihf", [ "armv8m.main", "thumb" ], FPV5_SP_D16>;
def CortexM35P : CortexMArchitectue<"cortex-m35p", "thumb2", "armv8m.main-none-eabihf", [ "armv8m.main", "thumb" ], FPV5_SP_D16>;
def CortexM4 : CortexMArchitectue<"cortex-m4", "thumb2", "armv7em-none-eabihf", [ "armv7m", "armv7em", "thumb" ], FPV4_SP_D16>;
def CortexM52 : CortexMArchitectue<"cortex-m52", "thumb2", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;
def CortexM55 : CortexMArchitectue<"cortex-m55", "thumb2", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;
def CortexM7 : CortexMArchitectue<"cortex-m7", "thumb2", "armv7em-none-eabihf", [ "armv7m", "armv7em", "thumb" ], FPV5_D16>;
def CortexM85 : CortexMArchitectue<"cortex-m85", "thumb2", "armv81m.main-none-eabihf", [ "armv81m.main", "thumb" ], FPV5_D16>;

#endif // _ARM_FAMILY_TD
