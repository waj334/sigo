include "base.td"

class UARTPeripheral<string Name, int Base> : Peripheral<Name, Base> {
  let accessWidth = 32;

  let registers = [
    Register<"CR",  0x00, 32, [
      Field<"EN", 0, 1, ReadWrite>,
      Field<"TXEIE", 7, 1, ReadWrite>
    ]>,

    Register<"SR",  0x04, 32, [
      Field<"TXE", 7, 1, Read>,
      Field<"RXNE", 5, 1, Read>
    ]>,

    Register<"DR",  0x08, 32, [
      Field<"DATA", 0, 8, ReadWrite>
    ]>
  ];
}

// Create instances of the UART peripheral at their respective addresses.
def UART1 : UARTPeripheral<"UART1", 0x40011000>;
def UART2 : UARTPeripheral<"UART2", 0x40004400>;
def UART3 : UARTPeripheral<"UART3", 0x40004800>;

// Create the peripheral group for each UART. A fixed-size array will be created from these.
def UART : PeripheralGroup<"UART", [ UART1, UART2, UART3 ]>;