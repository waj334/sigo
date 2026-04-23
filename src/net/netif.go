package net

/*
#include <lwip/netif.h>
#include <lwip/dhcp.h>
#include <lwip/ip_addr.h>
#include <lwip/etharp.h>
#include <netif/ethernet.h>
#include <string.h>

// Forward declarations for the Go-exported callbacks.
extern err_t go_netif_init(struct netif *netif);
extern err_t go_netif_linkoutput(struct netif *netif, struct pbuf *p);

// _go_netif_add allocates and registers a new netif. The state pointer is
// stored in netif->state so Go callbacks can recover the *NetInterface.
static inline struct netif* _go_netif_add(struct netif *n, void *state) {
    ip4_addr_t ipaddr, netmask, gw;
    IP4_ADDR(&ipaddr, 0, 0, 0, 0);
    IP4_ADDR(&netmask, 0, 0, 0, 0);
    IP4_ADDR(&gw, 0, 0, 0, 0);
    return netif_add(n, &ipaddr, &netmask, &gw, state, go_netif_init, ethernet_input);
}

static inline void _go_dhcp_start(struct netif *n) {
    dhcp_start(n);
}

// Helpers to set netif fields from Go.
static inline void _go_netif_set_hwaddr(struct netif *n, uint8_t *addr) {
    n->hwaddr_len = 6;
    memcpy(n->hwaddr, addr, 6);
}

static inline void _go_netif_set_mtu(struct netif *n, uint16_t mtu) {
    n->mtu = mtu;
}

static inline void _go_netif_set_flags(struct netif *n, uint8_t flags) {
    n->flags = flags;
}

static inline void _go_netif_set_output(struct netif *n) {
    n->output = etharp_output;
}

static inline void _go_netif_set_linkoutput(struct netif *n) {
    n->linkoutput = go_netif_linkoutput;
}

// Allocate a netif struct on the C heap so its lifetime is independent of any
// Go stack frame.
static inline struct netif* _go_netif_alloc(void) {
    return (struct netif*)malloc(sizeof(struct netif));
}
*/
import "C"
import "unsafe"

// init initialises and registers the LWIP netif for this interface. Must be
// called from the LWIP goroutine (via queueOperation).
func (ni *NetInterface) init() {
	// Allocate the C netif struct and register it with LWIP.
	raw := C._go_netif_alloc()
	n := (*netif)(C._go_netif_add(raw, unsafe.Pointer(ni)))
	ni.iface = n
	n.SetDefault()
	n.SetUp()
	n.SetLinkUp()
	C._go_dhcp_start(n.raw())
}

//go:export goNetifInit go_netif_init
func goNetifInit(n *C.netif) C.err_t {
	iface := (*netif)(n)
	ni := (*NetInterface)(iface.State())

	if ni == nil || ni.device == nil {
		return C.err_t(errLowLevelNetifError)
	}

	// Set the hardware address from the device.
	mac, err := ni.device.MACAddress()
	if err != nil {
		return C.err_t(errLowLevelNetifError)
	}
	C._go_netif_set_hwaddr(iface.raw(), (*C.uint8_t)(unsafe.Pointer(&mac[0])))
	C._go_netif_set_mtu(iface.raw(), 1500)

	flags := uint8(netifFlagBroadcast | netifFlagEthARP | netifFlagLinkUp | netifFlagEthernet)
	C._go_netif_set_flags(iface.raw(), C.uint8_t(flags))

	C._go_netif_set_output(iface.raw())
	C._go_netif_set_linkoutput(iface.raw())

	return C.err_t(errOk)
}

//go:export goNetifLinkoutput go_netif_linkoutput
func goNetifLinkoutput(n *C.netif, p *C.pbuf) C.err_t {
	iface := (*netif)(n)
	ni := (*NetInterface)(iface.State())

	if ni == nil || ni.device == nil {
		return C.err_t(errLowLevelNetifError)
	}

	// Linearize the pbuf chain into a Go byte slice.
	buf := (*packetBuffer)(p)
	totalLen := buf.TotalLen()
	frame := make([]byte, totalLen)
	buf.CopyPartial(unsafe.Pointer(&frame[0]), totalLen, 0)

	if err := ni.device.SendEthernet(frame); err != nil {
		return C.err_t(errLowLevelNetifError)
	}
	return C.err_t(errOk)
}

// IPAddress returns the current IPv4 address assigned to this interface.
// Returns [4]byte{0,0,0,0} if DHCP has not yet completed.
// Safe to call from any goroutine — reads a naturally-aligned uint32 on ARM.
func (ni *NetInterface) IPAddress() [4]byte {
	if ni.iface == nil {
		return [4]byte{}
	}
	return ni.iface.IP4Addr()
}

// feedFrame creates an LWIP pbuf from a raw Ethernet frame and feeds it into
// this interface's netif. Must be called from the LWIP goroutine.
func (ni *NetInterface) feedFrame(frame []byte) {
	if ni.iface == nil {
		return
	}

	p := newPacketBuffer(packetBufferLayerRaw, uint16(len(frame)), packetBufferTypePool)
	if p == nil {
		return
	}

	p.Take(unsafe.Pointer(&frame[0]), uint16(len(frame)))

	// Feed into the netif's input function (ethernet_input).
	ni.iface.Input(p)
}
