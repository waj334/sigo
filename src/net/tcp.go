package net

import "C"
import (
	"errors"
	"io"
	"time"
	"unsafe"
)

type TCPConn struct {
	// pcb points at lwIP's real tcp_pcb, allocated by tcpNew() and owned
	// by lwIP for the lifetime of the connection. Storing it by value
	// would freeze a snapshot of the pcb's state at dial time; subsequent
	// state machine transitions (SYN_SENT → ESTABLISHED → ...) happen on
	// lwIP's side and would never be visible through a Go-side copy.
	pcb         *tcpControlBlock
	rxBuf       chan []byte
	rxErr       chan error
	pending     []byte
	closed      bool
	localAddr   *TCPAddr
	remoteAddr  *TCPAddr
	connectDone chan error // signaled by the TCP connected/error callback
}

func (conn *TCPConn) Read(b []byte) (n int, err error) {
	// Drain any leftover from a previous partial read first.
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

func (conn *TCPConn) Write(b []byte) (n int, err error) {
	queueOperation(func() {
		remaining := b
		for len(remaining) > 0 {
			sndbuf := int(conn.pcb.SndBuf())
			if sndbuf == 0 {
				// The send buffer is full, so flush and let the caller retry.
				err = conn.pcb.Output()
				break
			}

			chunk := remaining
			if len(chunk) > sndbuf {
				chunk = remaining[:sndbuf]
			}

			err = conn.pcb.Write(chunk, tcpWriteFlagCopy)
			if err != nil {
				return
			}

			n += len(chunk)
			remaining = remaining[len(chunk):]
		}
		err = conn.pcb.Output()
	})
	return n, nil
}

func (conn *TCPConn) Close() error {
	var err error
	queueOperation(func() {
		if conn.pcb == nil {
			return
		}
		err = conn.pcb.Close()
		if err != nil {
			conn.pcb.Abort()
		}
		// lwIP frees the pcb on tcp_close/tcp_abort. Stop using it.
		conn.pcb = nil
		conn.closed = true
	})
	return err
}

func (conn *TCPConn) LocalAddr() Addr {
	if conn.localAddr != nil {
		return conn.localAddr
	}
	return nil
}

func (conn *TCPConn) RemoteAddr() Addr {
	if conn.remoteAddr != nil {
		return conn.remoteAddr
	}
	return nil
}

func (conn *TCPConn) SetDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

func (conn *TCPConn) SetReadDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

func (conn *TCPConn) SetWriteDeadline(t time.Time) error {
	return errors.New("net: deadlines not implemented")
}

//go:export tcpRecv tcp_recv_callback
func tcpRecv(arg unsafe.Pointer, pcb *tcpControlBlock, pbuf *packetBuffer, err lwipError) lwipError {
	conn := (*TCPConn)(arg)
	if pbuf == nil {
		conn.rxErr <- io.EOF
		return errOk
	}

	// Copy out of pbuf into Go-managed memory, then free the pbuf.
	buf := make([]byte, pbuf.TotalLen())
	pbuf.CopyPartial(unsafe.Pointer(&buf[0]), pbuf.TotalLen(), 0)
	pbuf.Free()

	// Acknowledge received bytes so lwIP opens the TCP window.
	pcb.Recved(pbuf.TotalLen())

	conn.rxBuf <- buf

	return errOk
}
