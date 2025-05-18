include "base.td"
include "arm/family.td"

def UARTPeripheral : PeripheralType<"UART", "The UART Peripheral"> {
  let accessWidth = 32;
  let registers = [
    Register<"CR",  0x00, 32, [
      Field<"EN", 0, 1, ReadWrite, "enable peripheral", [
        Enum<"enabled", 1, "enable the peripheral">,
        Enum<"disable", 0, "disable the peripheral">
      ]>,
      Field<"TXEIE", 7, 1, ReadWrite, "enable TX interrupt">
    ], "control register">,

    Register<"SR",  0x04, 32, [
      Field<"TXE", 7, 1, Read, "TX enabled status">,
      Field<"RXNE", 5, 1, Read, "RC enabled status">
    ], "status register">,

    Register<"DR",  0x0C, 32, [
      Field<"DATA", 0, 8, ReadWrite, "data available to read">
    ], "data register">
  ];
}

class UARTInstance<string Name, int Base> : PeripheralInstance<Name, Base, UARTPeripheral>;

// Create instances of the UART peripheral at their respective addresses.
def UART1 : UARTInstance<"UART1", 0x40011000>;
def UART2 : UARTInstance<"UART2", 0x40004400>;
def UART3 : UARTInstance<"UART3", 0x40004800>;

// Create the peripheral group for each UART. A fixed-size array will be created from these.
def UARTGroup : PeripheralGroup<[ UART1, UART2, UART3 ]>;