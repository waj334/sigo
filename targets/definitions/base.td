#ifndef _BASE_TD
#define _BASE_TD

class Object<string Name, string Description = ""> {
    string name = Name;
    string description = Description;
}

class StringValue<string Value> {
    string value = Value;
}

class FPU<string Value> : StringValue<Value>;
def NoFPU : FPU<"none">;
def SP : FPU<"SP">;
def DP : FPU<"DP">;

class MPU<string Value> : StringValue<Value>;
def NoMPU : MPU<"none">;
def MPUPresent : MPU<"MPU">;

class AccessMode<string Value> : StringValue<Value>;

def Read : AccessMode<"R">;
def Write : AccessMode<"W">;
def ReadWrite : AccessMode<"RW">;

class Family<string Name> : Object<Name>;
class Architecture<string Name, Family Fam, string Arch, string Triple, FPU Fpu = NoFPU, MPU Mpu = NoMPU> : Object<Name> {
    Family family = Fam;
    FPU fpu = Fpu;
    MPU mpu = Mpu;
    string triple = Triple;
    string arch = Arch;
}

class Quantity<int Value> {
    int value = Value;
}

class Bits<int Value> : Quantity<Value>;
class Bytes<int Value> : Quantity<Value>;

class Range<int Offset, int Width> {
  int offset = Offset;
  int width = Width;
}

class BitRange<int Offset, int Width> : Range<Offset, Width>;
class ByteRange<int Offset, int Width> : Range<Offset, Width>;
class MemoryRange<string Name, int Origin, int Size, string Description = "">
        : Object<Name, Description>, Range<Origin, Size>;

// Describes an acceptable value for a field.
class Enum<string Name, int Val, string Description = ""> : Object<Name, Description> {
    int value = Val;
}

// Describes a field within a register that a value can be read from and/or written to.
class Field<string Name, int Offset, int Width, AccessMode Access, string Description = "", list<Enum> Enums = []>
        : Object<Name, Description>, BitRange<Offset, Width> {
    AccessMode access = Access;
    list<Enum> enums = Enums;
}

// Describes a register containing fields that can be read from of written.
class Register<string Name, int Offset, int Width, list<Field> Fields = [], string Description = "">
        : Object<Name, Description>, ByteRange<Offset, Width> {
    list<Field> fields = Fields;
}

// Describes the prototype of a peripheral.
class PeripheralType<string Name, string Description = ""> : Object<Name, Description> {
    list<Register> registers;
    int accessWidth = 32;
}

class PeripheralInstance<string Name, int Base, PeripheralType Type, string Description = "">
        : Object<Name, Description> {
    int base = Base;
    PeripheralType type = Type;
}

// Describes a grouping of peripherals. Memboers of the group MUST be the same derived type.
class PeripheralGroup<list<PeripheralInstance> Instances, string Description = "">
        : Object<"Instances", Description> {
    list<PeripheralInstance> instances = Instances;
}

class Interrupt<string Name, int Line, string Description = ""> : Object<Name, Description> {
    int line = Line;
}

// Describes a variant within a chip series.
class Variant<string Name, string Description = ""> : Object<Name, Description> {
    list<MemoryRange> memories;
    list<Interrupt> interrupts;
}

class Series<string Name, string Description = ""> : Object<Name, Description> {
    Architecture arch;
    list<Variant> variants;
}

#endif // _BASE_TD
