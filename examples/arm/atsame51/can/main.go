//go:build atsame5x

package main

import (
	"peripheral/can"
	"peripheral/pin"
	"peripheral/uart"
	"runtime/arm/cortexm/sam/atsamx5x"
	"runtime/arm/cortexm/sam/atsamx5x/support/gclk"
	"runtime/arm/cortexm/sam/atsamx5x/support/mclk"
)

var (
	CAN        = can.CAN0
	UART       = uart.UART5
	messageBuf = make([]byte, can.FrameLengthInBytes(can.DLC8))
)

func main() {
	// Initialize the clock system
	atsamx5x.DefaultClocks()

	// Enable CAN0 in MCLK.
	mclk.Mclk.Ahbmask.SetCan0(true)
	for !mclk.Mclk.Intflag.GetCkrdy() {
	}

	// Configure the CAN0 peripheral clock to use GCLK1 (60MHz).
	gclk.Gclk.Pchctrl[atsamx5x.GCLK_CAN0].SetChen(false)
	for gclk.Gclk.Pchctrl[atsamx5x.GCLK_CAN0].GetChen() {
	}

	gclk.Gclk.Pchctrl[atsamx5x.GCLK_CAN0].SetGen(gclk.PchctrlGenGclk1)
	gclk.Gclk.Pchctrl[atsamx5x.GCLK_CAN0].SetChen(true)
	for !gclk.Gclk.Pchctrl[atsamx5x.GCLK_CAN0].GetChen() {
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
