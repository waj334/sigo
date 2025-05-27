#ifndef _BASE_TD
#define _BASE_TD

class Object<string Name, string Description = ""> {
    string name = Name;
    string description = Description;
}

class StringValue<string Value> {
    string value = Value;
}

class FPU<string Value, list<string> Features = []> : StringValue<Value> {
    list<string> features = Features;
}

def NoFPU : FPU<"none">;

class AccessMode<string Value> : StringValue<Value>;

def Read : AccessMode<"R">;
def Write : AccessMode<"W">;
def ReadWrite : AccessMode<"RW">;

class Family<string Name> : Object<Name>;
class Architecture<string Name, Family Fam, string Arch, string Triple, int Alignment = 4, FPU Fpu = NoFPU> : Object<Name> {
    Family family = Fam;
    FPU fpu = Fpu;
    string triple = Triple;
    string arch = Arch;
    int alignment = Alignment;
    list<string> runtimePackages = [];
    list<string> tags = [];
    list<string> features = [];
}

class Quantity<int Value, int Multiplier = 1> {
    int value = !mul(Value, Multiplier);
}

class Bit<int Value> : Quantity<Value>;

class MemoryUnit<int Value, int Multiplier> : Quantity<Value, Multiplier>;
class B<int Value> : MemoryUnit<Value, 1>;
class KB<int Value> : MemoryUnit<Value, 1000>;
class MB<int Value> : MemoryUnit<Value, 1000000>;
class GB<int Value> : MemoryUnit<Value, 1000000000>;

class Range<int Offset, int Width> {
  int offset = Offset;
  int width = Width;
}

class BitRange<int Offset, int Width> : Range<Offset, Width>;
class ByteRange<int Offset, int Width> : Range<Offset, Width>;
class MemoryRange<string Name, int Origin, MemoryUnit Size, bit Executable = false, AccessMode Access = ReadWrite,
    string Description = "">
        : Object<Name, Description>, Range<Origin, Size.value>
{
    AccessMode access = Access;
    bit executable = Executable;
}
defvar Executable = true;

// Describes an acceptable value for a field.
class Enum<string Name, int Val, string Description = ""> : Object<Name, Description> {
    int value = Val;
}

// Describes a field within a register that a value can be read from and/or written to.
class Field<string Name, int Offset, int Width, AccessMode Access, string Description = "", list<Enum> Enums = []>
        : Object<Name, Description>, BitRange<Offset, Width> {
    AccessMode access = Access;
    list<Enum> enums = Enums;
    int count = 1;
}

class RepeatingField<string Name, int Count, int Offset, int Width, AccessMode Access, string Description = "",
    list<Enum> Enums = []>
        : Field<Name, Offset, Width, Access, Description, Enums>
{
    let count = Count;
}

// Describes a register containing fields that can be read from of written.
class Register<string Name, int Offset, int Width, list<Field> Fields = [], string Description = "">
        : Object<Name, Description>, ByteRange<Offset, Width> {
    list<Field> fields = Fields;
    int count = 1;
}

class RepeatingRegister<string Name, int Count, int Offset, int Width, list<Field> Fields = [], string Description = "">
    : Register<Name, Offset, Width, Fields, Description>
{
    let count = Count;
}

// Describes the prototype of a peripheral.
class PeripheralType<string Name, string Description = ""> : Object<Name, Description> {
    list<Register> registers;
    int accessWidth = 32;
}

class PeripheralInstance<string Name, int Base, PeripheralType Type, string Description = "", list<string> Tags = []>
        : Object<Name, Description> {
    int base = Base;
    PeripheralType type = Type;
    list<string> tags = Tags;
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
class Variant<string Name, list<MemoryRange> Memories = [], list<Interrupt> Interrupts = [],
    MemoryUnit StackSize = KB<4>, list<string> Tags = [], string Description = "">
        : Object<Name, Description> {
    list<MemoryRange> memories = Memories;
    list<Interrupt> interrupts = Interrupts;
    int stackSize = StackSize.value;
    list<string> tags = Tags;
}

class Series<string Name, Architecture Arch, string Description = ""> : Object<Name, Description> {
    Architecture arch = Arch;
    list<Variant> variants = [];
    list<PeripheralType> peripheralTypes = [];
    list<string> runtimePackages = [];
    bit internal = false;
}

#endif // _BASE_TD
