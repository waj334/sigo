package net

import "C"
import (
	"errors"
	"time"
	"unsafe"
)

type UDPConn struct {
	pcb        udpControlBlock
	rxBuf      chan []byte
	rxErr      chan error
	pending    []byte
	closed     bool
	localAddr  *UDPAddr
	remoteAddr *UDPAddr
}

func (conn *UDPConn) Read(b []byte) (n int, err error) {
	if len(conn.pending) > 0 {
		n := copy(b, conn.pending)
		conn.pending = conn.pending[n:]
		return n, nil
	}

	select {
	case data := <-conn.rxBuf:
		n := copy(b, data)
		if n < len(data) {
			conn.pending = data[n:]
		}
		return n, nil
	case err := <-conn.rxErr:
		return 0, err
	}
}

func (conn *UDPConn) Write(b []byte) (n int, err error) {
	queueOperation(func() {
		err = conn.pcb.Send(b)
		if err == nil {
			n = len(b)
		}
	})
	return
}

func (conn *UDPConn) Close() error {
	var err error
	queueOperation(func() {
		conn.pcb.Remove()
		conn.closed = true
	})
	return err
}

func (conn *UDPConn) LocalAddr() Addr {
	if conn.localAddr != nil {
		return conn.localAddr
	}
	return nil
}

func (conn *UDPConn) RemoteAddr() Addr {
	if conn.remoteAddr != nil {
		return conn.remoteAddr
	}
	return nil
}

func (conn *UDPConn) SetDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

func (conn *UDPConn) SetReadDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

func (conn *UDPConn) SetWriteDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

//go:export udpRecvCallback udp_recv_callback
func udpRecvCallback(arg unsafe.Pointer, pcb *udpControlBlock, pbuf *packetBuffer, addr *ipAddr, port uint16) {
	conn := (*UDPConn)(arg)
	if pbuf == nil {
		return
	}

	buf := make([]byte, pbuf.TotalLen())
	pbuf.CopyPartial(unsafe.Pointer(&buf[0]), pbuf.TotalLen(), 0)
	pbuf.Free()

	conn.rxBuf <- buf
}
