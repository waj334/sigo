//go:build same51

package main

import (
	"peripheral/can"
	"peripheral/pin"
	"peripheral/uart"
	"runtime/arm/cortexm/sam/chip"
	"runtime/arm/cortexm/sam/samx51"
)

var (
	CAN        = can.CAN0
	UART       = uart.UART5
	messageBuf = make([]byte, can.FrameLengthInBytes(can.DLC8))
)

func main() {
	// Initialize the clock system
	samx51.DefaultClocks()

	// Enable CAN0 in MCLK.
	chip.MCLK.AHBMASK.SetCAN0(true)
	for !chip.MCLK.INTFLAG.GetCKRDY() {
	}

	// Configure the CAN0 peripheral clock to use GCLK1 (60MHz).
	chip.GCLK.PCHCTRL[samx51.GCLK_CAN0].SetCHEN(false)
	for chip.GCLK.PCHCTRL[samx51.GCLK_CAN0].GetCHEN() {
	}

	chip.GCLK.PCHCTRL[samx51.GCLK_CAN0].SetGEN(chip.GCLK_PCHCTRL_REG_GEN_GCLK1)
	chip.GCLK.PCHCTRL[samx51.GCLK_CAN0].SetCHEN(true)
	for !chip.GCLK.PCHCTRL[samx51.GCLK_CAN0].GetCHEN() {
	}

	// Configure UART
	UART.Configure(uart.Config{
		TXD:             pin.PB02,
		RXD:             pin.PB03,
		FrameFormat:     uart.UsartFrame,
		BaudHz:          115_200,
		CharacterSize:   8,
		NumStopBits:     1,
		ReceiveEnabled:  true,
		TransmitEnabled: true,
	})

	// Configure CAN0 for use with a 60MHz clock (GCLK1).
	if err := CAN.Configure(can.Config{
		TX:             pin.PA24,
		RX:             pin.PA23,
		TXQueueMode:    can.FIFOMode,
		TXNumElements:  1,
		TXDataLength:   can.DLC8,
		RX0NumElements: 1,
		RX0DataLength:  can.DLC8,
		StandardFilters: []can.Filter{
			{
				ID1:    1,
				ID2:    7,
				Type:   can.Range,
				Config: can.StoreInFIFO0,
			},
		},
		ExtendedFilters: []can.Filter{
			{
				ID1:    0xFF0,
				ID2:    0xFFF,
				Type:   can.Range,
				Config: can.StoreInFIFO0,
			},
		},
		StandardFilterMode: can.RejectAll,
		ExtendedFilterMode: can.RejectAll,
		FD:                 true,
		DataBitTiming: can.BitTiming{
			JumpWidth:         7,
			AfterSample:       7,
			BeforeSample:      20,
			Prescaler:         0,
			DelayCompensation: false,
		},
		NominalBitTiming: can.BitTiming{
			JumpWidth:    29,
			AfterSample:  29,
			BeforeSample: 88,
			Prescaler:    0,
		},
		OnNewMessage: func(fifo can.FIFO) {
			if frame, err := CAN.ReceiveFrame(fifo, messageBuf); err != nil {
				UART.WriteString(err.Error())
				UART.WriteString("\r\n")
			} else {
				// Echo the received frame.
				if err = CAN.SendFrame(frame); err != nil {
					UART.WriteString(err.Error())
					UART.WriteString("\r\n")
				}
			}
		},
	}); err != nil {
		UART.WriteString(err.Error())
		UART.WriteString("\r\n")
		panic(err)
	}

	for {
	}
}
