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

// Describes an acceptable value for a field.
class Value<string Name, int Val, string Description = ""> : Object<Name, Description> {
    int value = Val;
}

// Describes a field within a register that a value can be read from and/or written to.
class Field<string Name, int Offset, int Width, AccessMode Access, string Description = ""> : Object<Name, Description> {
    int offset = Offset;
    int width = Width;
    AccessMode access = Access;
    list<Value> values;
}

// Describes a register containing fields that can be read from of written.
class Register<string Name, int Offset, int Width, list<Field> Fields = [], string Description = "">
        : Object<Name, Description> {
    int offset = Offset;
    int width = Width;
    list<Field> fields = Fields;
}

// Describes a peripheral containing a set of registers.
class Peripheral<string Name, int Base, string Description = ""> : Object<Name, Description> {
    int base = Base;
    list<Register> registers;
    int accessWidth = 32;
}

// Describes a grouping of peripherals. Memboers of the group MUST be the same derived type.
class PeripheralGroup<string Name, list<Peripheral> Peripherals, string Description = ""> : Object<Name, Description> {
    list<Peripheral> peripherals = Peripherals;
}