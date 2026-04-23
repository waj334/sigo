package net

import (
	"errors"
	"time"
	"unsafe"
)

// Dialer contains options for connecting to an address.
type Dialer struct {
	// Timeout is the maximum amount of time a dial will wait for
	// a connect to complete.
	Timeout time.Duration

	// Deadline is the absolute point in time after which dials will fail.
	Deadline time.Time

	// LocalAddr is the local address to use when dialing.
	// If nil, a local address is automatically chosen.
	LocalAddr Addr

	// KeepAlive specifies the interval between keep-alive probes
	// for an active network connection.
	KeepAlive time.Duration
}

// Dial connects to the address on the named network.
//
// Known networks are "tcp", "tcp4", "udp", and "udp4".
//
// The address has the form "host:port".
// The host must be a literal IPv4 address.
func (d *Dialer) Dial(network, address string) (Conn, error) {
	switch network {
	case "tcp", "tcp4":
		raddr, err := ResolveTCPAddr(network, address)
		if err != nil {
			return nil, err
		}

		var laddr *TCPAddr
		if d.LocalAddr != nil {
			var ok bool
			laddr, ok = d.LocalAddr.(*TCPAddr)
			if !ok {
				return nil, errors.New("net: local address type mismatch")
			}
		}

		return d.DialTCP(network, laddr, raddr)
	case "udp", "udp4":
		raddr, err := ResolveUDPAddr(network, address)
		if err != nil {
			return nil, err
		}

		var laddr *UDPAddr
		if d.LocalAddr != nil {
			var ok bool
			laddr, ok = d.LocalAddr.(*UDPAddr)
			if !ok {
				return nil, errors.New("net: local address type mismatch")
			}
		}

		return d.DialUDP(network, laddr, raddr)
	default:
		return nil, errors.New("net: unknown network " + network)
	}
}

// DialTCP connects to the remote TCP address.
func (d *Dialer) DialTCP(network string, laddr, raddr *TCPAddr) (*TCPConn, error) {
	if raddr == nil {
		return nil, errors.New("net: nil remote address")
	}

	conn := &TCPConn{
		rxBuf:       make(chan []byte, 8),
		rxErr:       make(chan error, 1),
		connectDone: make(chan error, 1),
		remoteAddr:  raddr,
		localAddr:   laddr,
	}

	queueOperation(func() {
		pcb := tcpNew()
		if pcb == nil {
			conn.connectDone <- errors.New("net: failed to allocate TCP PCB")
			return
		}

		// Bind to local address if specified.
		if laddr != nil {
			addr := newIP4Addr(laddr.IP[0], laddr.IP[1], laddr.IP[2], laddr.IP[3])
			if err := pcb.Bind(&addr, uint16(laddr.Port)); err != nil {
				pcb.Abort()
				conn.connectDone <- errors.New("net: bind failed: " + err.Error())
				return
			}
		}

		conn.pcb = *pcb

		// Set the connection argument to point at our TCPConn.
		pcb.Arg(unsafe.Pointer(conn))

		// Install receive and error callbacks.
		pcb.SetRecv(tcpRecv)
		pcb.SetErr(tcpDialErr)

		// Initiate the TCP three-way handshake.
		rip := newIP4Addr(raddr.IP[0], raddr.IP[1], raddr.IP[2], raddr.IP[3])
		err := pcb.Connect(&rip, uint16(raddr.Port), tcpConnected)
		if err != nil {
			pcb.Abort()
			conn.connectDone <- errors.New("net: connect failed: " + err.Error())
			return
		}
	})

	// Wait for the connection callback or timeout.
	if d.Timeout > 0 {
		timer := time.NewTicker(d.Timeout)
		select {
		case err := <-conn.connectDone:
			timer.Stop()
			if err != nil {
				return nil, err
			}
		case <-timer.C:
			timer.Stop()
			queueOperation(func() {
				conn.pcb.Abort()
			})
			return nil, errors.New("net: dial timeout")
		}
	} else {
		if err := <-conn.connectDone; err != nil {
			return nil, err
		}
	}

	// Switch from the dial error callback to the normal connection error
	// callback now that the handshake is complete.
	queueOperation(func() {
		conn.pcb.SetErr(tcpConnErr)
	})

	return conn, nil
}

// DialUDP connects to the remote UDP address.
func (d *Dialer) DialUDP(network string, laddr, raddr *UDPAddr) (*UDPConn, error) {
	if raddr == nil {
		return nil, errors.New("net: nil remote address")
	}

	conn := &UDPConn{
		rxBuf:      make(chan []byte, 8),
		rxErr:      make(chan error, 1),
		remoteAddr: raddr,
		localAddr:  laddr,
	}

	var dialErr error
	queueOperation(func() {
		pcb := newUDPControlBlock()
		if pcb == nil {
			dialErr = errors.New("net: failed to allocate UDP PCB")
			return
		}

		if laddr != nil {
			addr := newIP4Addr(laddr.IP[0], laddr.IP[1], laddr.IP[2], laddr.IP[3])
			if err := pcb.Bind(&addr, uint16(laddr.Port)); err != nil {
				pcb.Remove()
				dialErr = errors.New("net: bind failed: " + err.Error())
				return
			}
		}

		conn.pcb = *pcb

		// Install receive callback.
		pcb.Recv(udpRecvCallback, unsafe.Pointer(conn))

		// Connect sets the default remote address for Send().
		rip := newIP4Addr(raddr.IP[0], raddr.IP[1], raddr.IP[2], raddr.IP[3])
		if err := pcb.Connect(&rip, uint16(raddr.Port)); err != nil {
			pcb.Remove()
			dialErr = errors.New("net: connect failed: " + err.Error())
			return
		}
	})

	if dialErr != nil {
		return nil, dialErr
	}

	return conn, nil
}

//go:export tcpConnected tcp_connected_callback
func tcpConnected(arg unsafe.Pointer, pcb *tcpControlBlock, err lwipError) lwipError {
	conn := (*TCPConn)(arg)
	if err != errOk {
		conn.connectDone <- errors.New("net: connect callback error: " + err.Error())
	} else {
		conn.connectDone <- nil
	}
	return errOk
}

// tcpDialErr is the error callback used during the TCP handshake phase.
//
//go:export tcpDialErr tcp_dial_err_callback
func tcpDialErr(arg unsafe.Pointer, err lwipError) {
	conn := (*TCPConn)(arg)
	conn.connectDone <- errors.New("net: connection error: " + err.Error())
}

// tcpConnErr is the error callback used for established connections.
//
//go:export tcpConnErr tcp_conn_err_callback
func tcpConnErr(arg unsafe.Pointer, err lwipError) {
	conn := (*TCPConn)(arg)
	conn.rxErr <- errors.New("net: connection error: " + err.Error())
}

// Dial connects to the address on the named network.
// See Dialer.Dial for details.
func Dial(network, address string) (Conn, error) {
	var d Dialer
	return d.Dial(network, address)
}

// DialTimeout acts like Dial but takes a timeout.
func DialTimeout(network, address string, timeout time.Duration) (Conn, error) {
	d := Dialer{Timeout: timeout}
	return d.Dial(network, address)
}
