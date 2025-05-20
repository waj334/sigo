include "base.td"
include "arm/family.td"
include "arm/cortexm/variant.td"

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

def ITMPeripheral : PeripheralType<"ITM", "Instrumentation Macrocell"> {
  let accessWidth = 32;
  let registers = [
    Register<"ITM_STIM", 0x0, 32, [
      Field<"STIMULUS", 0, 32, ReadWrite, "Data write to the stimulus port FIFO, for forwarding as a software event packet">,
      Field<"FIFOREADY", 0, 1, ReadWrite, "Indicates whether the stimulus port FIFO can accept data">,
    ], "Provides the interface for generating instrumentation messages">,
    Register<"ITM_TER", 0xe00, 32, [
      Field<"STIMENA", 0, 32, ReadWrite, "For bit STIMENA[n], in register ITM_TERx">,
    ], "Provide an individual enable bit for each ITM_STIM register">,
    Register<"ITM_TPR", 0xe40, 32, [
      Field<"PRIVMASK", 0, 32, ReadWrite, "Bit mask to enable unprivileged access to ITM stimulus ports">,
    ], "Controls which stimulus ports can be accessed by unprivileged code">,
    Register<"ITM_TCR", 0xe80, 32, [
      Field<"BUSY", 23, 1, ReadWrite, "Indicates whether the ITM is currently processing events">,
      Field<"TraceBusID", 16, 7, ReadWrite, "Identifier for multi-source trace stream formatting">,
      Field<"GTSFREQ", 10, 2, ReadWrite, "Global timestamp frequency">,
      Field<"TSPrescale", 8, 2, ReadWrite, "Local timestamp prescaler, used with the trace packet reference clock">,
      Field<"SWOENA", 4, 1, ReadWrite, "Enables asynchronous clocking of the timestamp counter">,
      Field<"TXENA", 3, 1, ReadWrite, "Enables forwarding of hardware event packet from the DWT unit to the ITM for output to the TPIU">,
      Field<"SYNCENA", 2, 1, ReadWrite, "Enables Synchronization packet transmission for a synchronous TPIU">,
      Field<"TSENA", 1, 1, ReadWrite, "Enables Local timestamp generation">,
      Field<"ITMENA", 0, 1, ReadWrite, "Enables the ITM">,
    ], "Configures and controls transfers through the ITM interface">,
  ];
}

def ITM : PeripheralInstance<"ITM", 0xE0000000, ITMPeripheral>;

class UARTInstance<string Name, int Base> : PeripheralInstance<Name, Base, UARTPeripheral>;

// Create instances of the UART peripheral at their respective addresses.
def UART1 : UARTInstance<"UART1", 0x40011000>;
def UART2 : UARTInstance<"UART2", 0x40004400>;
def UART3 : UARTInstance<"UART3", 0x40004800>;

// Create the peripheral group for each UART. A fixed-size array will be created from these.
def UARTGroup : PeripheralGroup<[ UART1, UART2, UART3 ]>;

def STM32H : Series<"STM32H", CortexM7> {
    let variants = [
        CortexMVariant<"STM32H747XI", [
            MemoryRange<"ITCRAM", 0x0, KB<64>>,
            MemoryRange<"FLASH", 0x08000000, KB<2048>, Executable>,
            MemoryRange<"RAM", 0x24000000, KB<1024>>,
        ], [
            Interrupt<"External0", 0>,
            Interrupt<"External1", 1, "the description">,
            Interrupt<"External2", 2>,
        ]>,
    ];
}