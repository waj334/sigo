class AccessMode<string Value> {
    string value = Value;
}

def Read : AccessMode<"R">;
def Write : AccessMode<"W">;
def ReadWrite : AccessMode<"RW">;

class Object<string Name, string Description = ""> {
    string name = Name;
    string description = Description;
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