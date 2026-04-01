package net

// #cgo LDFLAGS: -llwip
/*
#include <lwip/init.h>
#include <lwip/ip_addr.h>
#include <lwip/netif.h>
#include <lwip/pbuf.h>
#include <lwip/tcp.h>
#include <lwip/timeouts.h>
#include <lwip/udp.h>
#include <stdint.h>
#include <stdbool.h>

// Accessor helpers for struct pbuf fields (avoids needing CIR struct field access).
static inline void*     _go_pbuf_next(struct pbuf *p)    { return p->next; }
static inline void*     _go_pbuf_payload(struct pbuf *p)  { return p->payload; }
static inline uint16_t  _go_pbuf_tot_len(struct pbuf *p)  { return p->tot_len; }
static inline uint16_t  _go_pbuf_len(struct pbuf *p)      { return p->len; }
static inline uint8_t   _go_pbuf_type(struct pbuf *p)     { return p->type_internal; }
static inline uint8_t   _go_pbuf_flags(struct pbuf *p)    { return p->flags; }
static inline uint16_t  _go_pbuf_ref(struct pbuf *p)      { return (uint16_t)p->ref; }

// pbuf_layer / pbuf_type enum values. Enum values are rvalues and cannot appear
// in the __sigo_cgo_refs address array, so expose them as callable functions.
static inline int32_t _go_pbuf_layer_transport() { return (int32_t)PBUF_TRANSPORT; }
static inline int32_t _go_pbuf_layer_ip()        { return (int32_t)PBUF_IP; }
static inline int32_t _go_pbuf_layer_link()      { return (int32_t)PBUF_LINK; }
static inline int32_t _go_pbuf_layer_raw_tx()    { return (int32_t)PBUF_RAW_TX; }
static inline int32_t _go_pbuf_layer_raw()       { return (int32_t)PBUF_RAW; }
static inline int32_t _go_pbuf_type_ram()        { return (int32_t)PBUF_RAM; }
static inline int32_t _go_pbuf_type_rom()        { return (int32_t)PBUF_ROM; }
static inline int32_t _go_pbuf_type_ref()        { return (int32_t)PBUF_REF; }
static inline int32_t _go_pbuf_type_pool()       { return (int32_t)PBUF_POOL; }

// Accessor helpers for struct netif fields.
static inline void*     _go_netif_get_state(struct netif *n)   { return n->state; }
static inline uint16_t  _go_netif_mtu(struct netif *n)        { return n->mtu; }
static inline uint8_t   _go_netif_flags_get(struct netif *n)   { return n->flags; }
static inline uint8_t   _go_netif_hwaddr_len(struct netif *n)  { return n->hwaddr_len; }
static inline uint8_t*  _go_netif_hwaddr(struct netif *n)      { return n->hwaddr; }
static inline uint8_t   _go_netif_num(struct netif *n)         { return n->num; }
static inline char      _go_netif_name0(struct netif *n)       { return n->name[0]; }
static inline char      _go_netif_name1(struct netif *n)       { return n->name[1]; }
static inline uint32_t  _go_netif_ip4_addr(struct netif *n)    { return netif_ip4_addr(n)->addr; }

// netif status check macros exposed as functions.
static inline uint8_t _go_netif_is_up(struct netif *n)       { return netif_is_up(n); }
static inline uint8_t _go_netif_is_link_up(struct netif *n)  { return netif_is_link_up(n); }

// NETIF_FLAG constants exposed as functions for CGo.
static inline uint8_t _go_netif_flag_up()         { return NETIF_FLAG_UP; }
static inline uint8_t _go_netif_flag_broadcast()   { return NETIF_FLAG_BROADCAST; }
static inline uint8_t _go_netif_flag_etharp()       { return NETIF_FLAG_ETHARP; }
static inline uint8_t _go_netif_flag_link_up()      { return NETIF_FLAG_LINK_UP; }
static inline uint8_t _go_netif_flag_ethernet()     { return NETIF_FLAG_ETHERNET; }
static inline uint8_t _go_netif_flag_igmp()         { return NETIF_FLAG_IGMP; }
static inline uint8_t _go_netif_flag_mld6()         { return NETIF_FLAG_MLD6; }

// Backlog wrappers: expand correctly whether TCP_LISTEN_BACKLOG is set or not.
static inline void _go_tcp_backlog_delayed(struct tcp_pcb *pcb)  { tcp_backlog_delayed(pcb); }
static inline void _go_tcp_backlog_accepted(struct tcp_pcb *pcb) { tcp_backlog_accepted(pcb); }

static inline uint16_t _go_tcp_mss(struct tcp_pcb *pcb) { return tcp_mss(pcb); }
static inline uint16_t _go_tcp_sndbuf(struct tcp_pcb *pcb) { return tcp_sndbuf(pcb); }
static inline uint16_t _go_tcp_sndqueuelen(struct tcp_pcb *pcb) { return tcp_sndqueuelen(pcb); }
static inline void _go_tcp_nagle_disable(struct tcp_pcb *pcb) { tcp_nagle_disable(pcb); }
static inline void _go_tcp_nagle_enable(struct tcp_pcb *pcb) { tcp_nagle_enable(pcb); }
static inline bool _go_tcp_nagle_disabled(struct tcp_pcb *pcb) { return tcp_nagle_disabled(pcb); }
static inline struct tcp_pcb * _go_tcp_listen(struct tcp_pcb *pcb) { return tcp_listen(pcb); }

// Build an ip_addr_t from four octets (IPv4).
static inline ip_addr_t _go_ip4_addr(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    ip_addr_t addr;
    IP4_ADDR(&addr, a, b, c, d);
    return addr;
}

static inline uint8_t _go_lwip_ip_addr_type_IPADDR_TYPE_V4() { return IPADDR_TYPE_V4; }
static inline uint8_t _go_lwip_ip_addr_type_IPADDR_TYPE_V6() { return IPADDR_TYPE_V6; }
static inline uint8_t _go_lwip_ip_addr_type_IPADDR_TYPE_ANY() { return IPADDR_TYPE_ANY; }

*/
import "C"
import (
	"unsafe"
)

// lwipError matches err_t = s8_t = int8_t.
type lwipError int8

func (e lwipError) Error() string {
	switch e {
	case errOk:
		return "No error, everything OK"
	case errMem:
		return "Out of memory error"
	case errBuf:
		return "Buffer error"
	case errTimeout:
		return "Timeout"
	case errRoute:
		return "Routing problem"
	case errInProgress:
		return "Operation in progres"
	case errValue:
		return "Illegal value"
	case errWouldBlock:
		return "Operation would block"
	case errAddressInUse:
		return "Address in use"
	case errAlreadyConnecting:
		return "Already connecting"
	case errConnectionAlreadyEstablished:
		return "Conn already established"
	case errNotConnected:
		return "Not connected"
	case errLowLevelNetifError:
		return "Low-level netif erro"
	case errConnectionAborted:
		return "Connection aborted"
	case errConnectionReset:
		return "Connection reset"
	case errConnectionClosed:
		return "Connection closed"
	case errIllegalArgument:
		return "Illegal argument"
	default:
		return "unknown lwip error"
	}
}

func unwrapLwipError(err lwipError) error {
	if err == errOk {
		return nil
	}
	return err
}

// err_t values are a C enum; hardcode them to avoid C.ERR_* references
// (enum values are rvalues and cannot be addressed by __sigo_cgo_refs).
const (
	errOk                           = lwipError(0)   // No error, everything OK.
	errMem                          = lwipError(-1)  // Out of memory error.
	errBuf                          = lwipError(-2)  // Buffer error.
	errTimeout                      = lwipError(-3)  // Timeout.
	errRoute                        = lwipError(-4)  // Routing problem.
	errInProgress                   = lwipError(-5)  // Operation in progress.
	errValue                        = lwipError(-6)  // Illegal value.
	errWouldBlock                   = lwipError(-7)  // Operation would block.
	errAddressInUse                 = lwipError(-8)  // Address in use.
	errAlreadyConnecting            = lwipError(-9)  // Already connecting.
	errConnectionAlreadyEstablished = lwipError(-10) // Conn already established.
	errNotConnected                 = lwipError(-11) // Not connected.
	errLowLevelNetifError           = lwipError(-12) // Low-level netif error.
	errConnectionAborted            = lwipError(-13) // Connection aborted.
	errConnectionReset              = lwipError(-14) // Connection reset.
	errConnectionClosed             = lwipError(-15) // Connection closed.
	errIllegalArgument              = lwipError(-16) // Illegal argument.
)

type packetBufferLayer uint16
type packetBufferType uint16

// pbuf_layer and pbuf_type are enums whose values depend on lwip configuration.
// Use C wrapper functions so the values are resolved at link time.
var (
	packetBufferLayerTransport = packetBufferLayer(C._go_pbuf_layer_transport()) // Spare room for transport layer header (UDP/TCP).
	packetBufferLayerIp        = packetBufferLayer(C._go_pbuf_layer_ip())        // Spare room for IP header.
	packetBufferLayerLink      = packetBufferLayer(C._go_pbuf_layer_link())      // Spare room for link layer header (Ethernet).
	packetBufferLayerRawTx     = packetBufferLayer(C._go_pbuf_layer_raw_tx())    // Spare room for encapsulation header before Ethernet.
	packetBufferLayerRaw       = packetBufferLayer(C._go_pbuf_layer_raw())       // For input packets in a netif driver.

	packetBufferTypeRam  = packetBufferType(C._go_pbuf_type_ram())  // Data in RAM; use for TX.
	packetBufferTypeRom  = packetBufferType(C._go_pbuf_type_rom())  // Data in ROM.
	packetBufferTypeRef  = packetBufferType(C._go_pbuf_type_ref())  // From pbuf pool; volatile payload.
	packetBufferTypePool = packetBufferType(C._go_pbuf_type_pool()) // From pool; use for RX.
)

// TCP write API flags (tcp.h defines; hardcoded since #define values are not addressable).
const (
	tcpWriteFlagCopy = uint8(0x01) // Data is copied into internal buffers.
	tcpWriteFlagMore = uint8(0x02) // More data to come; do not push yet.
)

type ipAddrType uint8

const (
	ipAddrTypeV4  = 0
	ipAddrTypeV6  = 6
	ipAddrTypeAny = 46
)

type ipAddr C.ip_addr_t

func (addr *ipAddr) raw() *C.ip_addr_t {
	return (*C.ip_addr_t)(addr)
}

func newIP4Addr(a, b, c, d byte) ipAddr {
	return ipAddr(C._go_ip4_addr(a, b, c, d))
}

type netif C.netif

func (n *netif) raw() *C.netif {
	return (*C.netif)(n)
}

// Netif flag constants. Values come from C macros resolved at link time.
type netifFlag uint8

var (
	netifFlagUp        = netifFlag(C._go_netif_flag_up())
	netifFlagBroadcast = netifFlag(C._go_netif_flag_broadcast())
	netifFlagEthARP    = netifFlag(C._go_netif_flag_etharp())
	netifFlagLinkUp    = netifFlag(C._go_netif_flag_link_up())
	netifFlagEthernet  = netifFlag(C._go_netif_flag_ethernet())
	netifFlagIGMP      = netifFlag(C._go_netif_flag_igmp())
	netifFlagMLD6      = netifFlag(C._go_netif_flag_mld6())
)

func (n *netif) SetDefault() {
	C.netif_set_default(n.raw())
}

func (n *netif) SetUp() {
	C.netif_set_up(n.raw())
}

func (n *netif) SetDown() {
	C.netif_set_down(n.raw())
}

func (n *netif) SetLinkUp() {
	C.netif_set_link_up(n.raw())
}

func (n *netif) SetLinkDown() {
	C.netif_set_link_down(n.raw())
}

func (n *netif) Remove() {
	C.netif_remove(n.raw())
}

func (n *netif) IsUp() bool {
	return C._go_netif_is_up(n.raw()) != 0
}

func (n *netif) IsLinkUp() bool {
	return C._go_netif_is_link_up(n.raw()) != 0
}

func (n *netif) MTU() uint16 {
	return uint16(C._go_netif_mtu(n.raw()))
}

func (n *netif) Flags() netifFlag {
	return netifFlag(C._go_netif_flags_get(n.raw()))
}

func (n *netif) HWAddrLen() uint8 {
	return uint8(C._go_netif_hwaddr_len(n.raw()))
}

func (n *netif) HWAddr() []byte {
	length := n.HWAddrLen()
	ptr := C._go_netif_hwaddr(n.raw())
	// Build a slice from the C array without copying.
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr)), length)
}

func (n *netif) Num() uint8 {
	return uint8(C._go_netif_num(n.raw()))
}

func (n *netif) Name() string {
	return string([]byte{byte(C._go_netif_name0(n.raw())), byte(C._go_netif_name1(n.raw()))})
}

func (n *netif) IP4Addr() [4]byte {
	addr := uint32(C._go_netif_ip4_addr(n.raw()))
	return [4]byte{byte(addr), byte(addr >> 8), byte(addr >> 16), byte(addr >> 24)}
}

func (n *netif) Input(p *packetBuffer) lwipError {
	return lwipError(C.netif_input(p.raw(), n.raw()))
}

func (n *netif) State() unsafe.Pointer {
	return C._go_netif_get_state(n.raw())
}

// NetInterface pairs a LWIP netif with its backing network device and
// per-interface receive channel. Multiple NetInterfaces can coexist.
type NetInterface struct {
	iface  *netif
	device NetDevice
	rx     chan []byte
}

// Netif returns the underlying LWIP netif.
func (ni *NetInterface) Netif() *netif {
	return ni.iface
}

// Device returns the backing NetDevice.
func (ni *NetInterface) Device() NetDevice {
	return ni.device
}

// TCP callback function types matching the C callback signatures.
type (
	tcpAcceptFn    func(arg unsafe.Pointer, newpcb *tcpControlBlock, err lwipError) lwipError
	tcpRecvFn      func(arg unsafe.Pointer, pcb *tcpControlBlock, p *packetBuffer, err lwipError) lwipError
	tcpSentFn      func(arg unsafe.Pointer, pcb *tcpControlBlock, len uint16) lwipError
	tcpPollFn      func(arg unsafe.Pointer, pcb *tcpControlBlock) lwipError
	tcpErrFn       func(arg unsafe.Pointer, err lwipError)
	tcpConnectedFn func(arg unsafe.Pointer, pcb *tcpControlBlock, err lwipError) lwipError
)

func lwipInit() {
	C.lwip_init()
}

func lwipSysCheckTimeouts() {
	C.sys_check_timeouts()
}

func tcpNew() *tcpControlBlock {
	return (*tcpControlBlock)(C.tcp_new())
}

func tcpNewIpType(ipType uint8) *tcpControlBlock {
	return (*tcpControlBlock)(C.tcp_new_ip_type(ipType))
}

type packetBuffer C.pbuf

func newPacketBuffer(layer packetBufferLayer, length uint16, typ packetBufferType) *packetBuffer {
	return (*packetBuffer)(C.pbuf_alloc(int32(layer), length, int32(typ)))
}

func newPacketBufferCustom(layer packetBufferLayer, length uint16, typ packetBufferType, custom *C.pbuf_custom, payloadMem unsafe.Pointer, payloadMemLen uint16) *packetBuffer {
	return (*packetBuffer)(C.pbuf_alloced_custom(int32(layer), length, int32(typ), custom, payloadMem, payloadMemLen))
}

func newPacketBufferFrom(b []byte, typ packetBufferType) *packetBuffer {
	return (*packetBuffer)(C.pbuf_alloc_reference(unsafe.Pointer(&b[0]), uint16(len(b)), int32(typ)))
}

func (p *packetBuffer) raw() *C.pbuf {
	return (*C.pbuf)(p)
}

func (p *packetBuffer) Next() *packetBuffer {
	return (*packetBuffer)(C._go_pbuf_next(p.raw()))
}

func (p *packetBuffer) Payload() unsafe.Pointer {
	return C._go_pbuf_payload(p.raw())
}

func (p *packetBuffer) TotalLen() uint16 {
	return uint16(C._go_pbuf_tot_len(p.raw()))
}

func (p *packetBuffer) Len() uint16 {
	return uint16(C._go_pbuf_len(p.raw()))
}

func (p *packetBuffer) Type() packetBufferType {
	return packetBufferType(C._go_pbuf_type(p.raw()))
}

func (p *packetBuffer) Flags() uint8 {
	return uint8(C._go_pbuf_flags(p.raw()))
}

func (p *packetBuffer) RefCount() uint16 {
	return uint16(C._go_pbuf_ref(p.raw()))
}

func (p *packetBuffer) Realloc(length uint16) {
	C.pbuf_realloc(p.raw(), length)
}

func (p *packetBuffer) Free() uint8 {
	return uint8(C.pbuf_free(p.raw()))
}

func (p *packetBuffer) Ref() {
	C.pbuf_ref(p.raw())
}

func (p *packetBuffer) Cat(other *packetBuffer) {
	C.pbuf_cat(p.raw(), other.raw())
}

func (p *packetBuffer) Chain(other *packetBuffer) {
	C.pbuf_chain(p.raw(), other.raw())
}

func (p *packetBuffer) Copy(other *packetBuffer) lwipError {
	return lwipError(C.pbuf_copy(p.raw(), other.raw()))
}

func (p *packetBuffer) CopyPartial(data unsafe.Pointer, length uint16, offset uint16) uint16 {
	return uint16(C.pbuf_copy_partial(p.raw(), data, length, offset))
}

func (p *packetBuffer) Skip(offset uint16) (*packetBuffer, uint16) {
	var outOffset uint16
	return (*packetBuffer)(C.pbuf_skip(p.raw(), offset, &outOffset)), outOffset
}

func (p *packetBuffer) Take(data unsafe.Pointer, length uint16) lwipError {
	return lwipError(C.pbuf_take(p.raw(), data, length))
}

func (p *packetBuffer) TakeAt(data unsafe.Pointer, length uint16, offset uint16) lwipError {
	return lwipError(C.pbuf_take_at(p.raw(), data, length, offset))
}

func (p *packetBuffer) Coalesce(layer packetBufferLayer) *packetBuffer {
	return (*packetBuffer)(C.pbuf_coalesce(p.raw(), int32(layer)))
}

func (p *packetBuffer) GetAt(offset uint16) uint8 {
	return uint8(C.pbuf_get_at(p.raw(), offset))
}

func (p *packetBuffer) TryGetAt(offset uint16) int {
	return int(C.pbuf_try_get_at(p.raw(), offset))
}

func (p *packetBuffer) PutAt(offset uint16, data uint8) {
	C.pbuf_put_at(p.raw(), offset, data)
}

func (p *packetBuffer) Memcmp(offset uint16, data unsafe.Pointer, length uint16) uint16 {
	return uint16(C.pbuf_memcmp(p.raw(), offset, data, length))
}

func (p *packetBuffer) Memfind(data unsafe.Pointer, length uint16, startOffset uint16) uint16 {
	return uint16(C.pbuf_memfind(p.raw(), data, length, startOffset))
}

type tcpControlBlock C.tcp_pcb

func (pcb *tcpControlBlock) raw() *C.tcp_pcb {
	return (*C.tcp_pcb)(pcb)
}

func (pcb *tcpControlBlock) BacklogDelayed() {
	C._go_tcp_backlog_delayed(pcb.raw())
}

func (pcb *tcpControlBlock) BacklogAccepted() {
	C._go_tcp_backlog_accepted(pcb.raw())
}

func (pcb *tcpControlBlock) Close() error {
	return unwrapLwipError(
		lwipError(C.tcp_close(pcb.raw())))
}

func (pcb *tcpControlBlock) Shutdown(shutRead, shutWrite bool) lwipError {
	var rx, tx int32
	if shutRead {
		rx = 1
	}
	if shutWrite {
		tx = 1
	}
	return lwipError(C.tcp_shutdown(pcb.raw(), rx, tx))
}

func (pcb *tcpControlBlock) Abort() {
	C.tcp_abort(pcb.raw())
}

func (pcb *tcpControlBlock) Bind(ipaddr *ipAddr, port uint16) error {
	return unwrapLwipError(
		lwipError(C.tcp_bind(pcb.raw(), ipaddr.raw(), port)))
}

func (pcb *tcpControlBlock) ListenWithBacklog(backlog uint8) *tcpControlBlock {
	return (*tcpControlBlock)(C.tcp_listen_with_backlog(pcb.raw(), backlog))
}

func (pcb *tcpControlBlock) ListenWithBacklogAndErr(backlog uint8) (*tcpControlBlock, error) {
	var err int8
	return (*tcpControlBlock)(C.tcp_listen_with_backlog_and_err(pcb.raw(), backlog, &err)),
		unwrapLwipError(lwipError(err))
}

func (pcb *tcpControlBlock) Recved(length uint16) {
	C.tcp_recved(pcb.raw(), length)
}

func (pcb *tcpControlBlock) Arg(arg unsafe.Pointer) {
	C.tcp_arg(pcb.raw(), arg)
}

func (pcb *tcpControlBlock) SetRecv(fn tcpRecvFn) {
	C.tcp_recv(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)))
}

func (pcb *tcpControlBlock) SetSent(fn tcpSentFn) {
	C.tcp_sent(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)))
}

func (pcb *tcpControlBlock) SetErr(fn tcpErrFn) {
	C.tcp_err(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)))
}

func (pcb *tcpControlBlock) SetAccept(fn tcpAcceptFn) {
	C.tcp_accept(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)))
}

func (pcb *tcpControlBlock) SetPoll(fn tcpPollFn, interval uint8) {
	C.tcp_poll(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)), interval)
}

func (pcb *tcpControlBlock) Connect(ipAddr *ipAddr, port uint16, connected tcpConnectedFn) error {
	return unwrapLwipError(
		lwipError(C.tcp_connect(pcb.raw(), ipAddr.raw(), port, *(*unsafe.Pointer)(unsafe.Pointer(&connected)))))
}

func (pcb *tcpControlBlock) Write(data []byte, apiflags uint8) error {
	return unwrapLwipError(
		lwipError(C.tcp_write(pcb.raw(), unsafe.Pointer(&data[0]), uint16(len(data)), apiflags)))
}

func (pcb *tcpControlBlock) Output() lwipError {
	return lwipError(C.tcp_output(pcb.raw()))
}

func (pcb *tcpControlBlock) SetPrio(prio uint8) {
	C.tcp_setprio(pcb.raw(), prio)
}

func (pcb *tcpControlBlock) BindNetif(n *netif) {
	C.tcp_bind_netif(pcb.raw(), n.raw())
}

func (pcb *tcpControlBlock) Mss() uint16 {
	return uint16(C._go_tcp_mss(pcb.raw()))
}

func (pcb *tcpControlBlock) SndBuf() uint16 {
	return uint16(C._go_tcp_sndbuf(pcb.raw()))
}

func (pcb *tcpControlBlock) SndQueueLen() uint16 {
	return uint16(C._go_tcp_sndqueuelen(pcb.raw()))
}

func (pcb *tcpControlBlock) NagleDisable() {
	C._go_tcp_nagle_disable(pcb.raw())
}

func (pcb *tcpControlBlock) NagleEnable() {
	C._go_tcp_nagle_enable(pcb.raw())
}

func (pcb *tcpControlBlock) NagleDisabled() bool {
	return bool(C._go_tcp_nagle_disabled(pcb.raw()))
}

func (pcb *tcpControlBlock) Listen() *tcpControlBlock {
	return (*tcpControlBlock)(C._go_tcp_listen(pcb.raw()))
}

type (
	udpRecvFn func(arg unsafe.Pointer, pcb *udpControlBlock, p *packetBuffer, addr *ipAddr, port uint16)
)

/*
typedef void (*udp_recv_fn)(void *arg, struct udp_pcb *pcb, struct pbuf *p,
    const ip_addr_t *addr, u16_t port);
*/

type udpControlBlock C.udp_pcb

func newUDPControlBlock() *udpControlBlock {
	return (*udpControlBlock)(C.udp_new())
}

func newUDPControlBlockIpType(ipType ipAddrType) *udpControlBlock {
	return (*udpControlBlock)(C.udp_new_ip_type(uint8(ipType)))
}

func (pcb *udpControlBlock) raw() *C.udp_pcb {
	return (*C.udp_pcb)(pcb)
}

func (pcb *udpControlBlock) Remove() {
	C.udp_remove(pcb.raw())
}

func (pcb *udpControlBlock) Bind(ipaddr *ipAddr, port uint16) error {
	return unwrapLwipError(
		lwipError(C.udp_bind(pcb.raw(), ipaddr.raw(), port)))
}

func (pcb *udpControlBlock) BindNetif(n *netif) {
	C.udp_bind_netif(pcb.raw(), n.raw())
}

func (pcb *udpControlBlock) Connect(ipaddr *ipAddr, port uint16) error {
	return unwrapLwipError(
		lwipError(C.udp_connect(pcb.raw(), ipaddr.raw(), port)))
}

func (pcb *udpControlBlock) Disconnect() {
	C.udp_disconnect(pcb.raw())
}

func (pcb *udpControlBlock) Recv(fn udpRecvFn, arg unsafe.Pointer) {
	C.udp_recv(pcb.raw(), *(*unsafe.Pointer)(unsafe.Pointer(&fn)), arg)
}

func (pcb *udpControlBlock) SendToIf(data []byte, dest *ipAddr, destPort uint16, netif *netif) error {
	p := newPacketBufferFrom(data, packetBufferTypeRef)
	err := unwrapLwipError(
		lwipError(C.udp_sendto_if(pcb.raw(), p.raw(), dest.raw(), destPort, netif.raw())))
	p.Free()
	return err
}

func (pcb *udpControlBlock) SendToIfSrc(data []byte, dest *ipAddr, destPort uint16, netif *netif, srcIp *ipAddr) error {
	p := newPacketBufferFrom(data, packetBufferTypeRef)
	err := unwrapLwipError(
		lwipError(C.udp_sendto_if_src(pcb.raw(), p.raw(), dest.raw(), destPort, netif.raw(), srcIp.raw())))
	p.Free()
	return err
}

func (pcb *udpControlBlock) Sendto(data []byte, ipaddr *ipAddr, port uint16) error {
	p := newPacketBufferFrom(data, packetBufferTypeRef)
	err := unwrapLwipError(
		lwipError(C.udp_sendto(pcb.raw(), p.raw(), ipaddr.raw(), uint16(port))))
	p.Free()
	return err
}

func (pcb *udpControlBlock) Send(data []byte) error {
	p := newPacketBufferFrom(data, packetBufferTypeRef)
	err := unwrapLwipError(
		lwipError(C.udp_send(pcb.raw(), p.raw())))
	p.Free()
	return err
}
