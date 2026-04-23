package net

import (
	"fmt"
	"os"
)

// NetDevice is the interface a network driver must implement to integrate
// with the LWIP-based net package.
type NetDevice interface {
	// SendEthernet transmits a raw Ethernet frame (dest MAC | src MAC | ethertype | payload).
	SendEthernet(frame []byte) error

	// MACAddress returns the 6-byte hardware address of the device.
	MACAddress() ([6]byte, error)

	// SetRxCallback registers a function the driver must call when an Ethernet
	// frame is received on the data channel.
	SetRxCallback(fn func(frame []byte))
}

// interfaces holds all registered NetInterfaces. Only mutated from the LWIP
// goroutine (via queueOperation), so no synchronization is needed.
var interfaces []*NetInterface

// RegisterNetDevice creates a LWIP netif backed by dev, brings it up, and
// starts DHCP. Safe to call multiple times for different devices. Returns the
// *NetInterface so the caller can inspect or further configure it.
func RegisterNetDevice(dev NetDevice) *NetInterface {
	ni := &NetInterface{
		device: dev,
		rx:     make(chan []byte, 8),
	}

	// Register RX callback with the driver. Non-blocking send drops frames
	// if the channel is full to avoid deadlocking the driver's poll loop.
	dev.SetRxCallback(func(frame []byte) {
		// Copy the frame so the driver can reuse its buffer.
		buf := make([]byte, len(frame))
		copy(buf, frame)
		select {
		case ni.rx <- buf:
		default:
			fmt.Fprintf(os.Stdout, "[NET] RX drop (ch full), len=%d\n", len(frame))
		}
	})

	// Initialize the LWIP netif on the LWIP goroutine.
	queueOperation(func() {
		ni.init()
		interfaces = append(interfaces, ni)
	})

	return ni
}
