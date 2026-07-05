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
	readErr     error // latched terminal read error (EOF or transport failure)
	closed      bool
	localAddr   *TCPAddr
	remoteAddr  *TCPAddr
	connectDone chan error // signaled by the TCP connected/error callback

	// sendSpace is signaled (capacity 1, never blocking) by the tcp_sent
	// callback when the peer ACKs data and by the error callback when the
	// connection dies. Write waits on it when the send buffer is full.
	sendSpace chan struct{}
}

func (conn *TCPConn) Read(b []byte) (n int, err error) {
	// Drain any leftover from a previous partial read first.
	if len(conn.pending) > 0 {
		n := copy(b, conn.pending)
		conn.pending = conn.pending[n:]
		return n, nil
	}

	// A terminal error is latched: without this, a second Read after EOF
	// would block forever on the empty channels below.
	if conn.readErr != nil {
		return 0, conn.readErr
	}

	// Buffered data and a connection error (typically EOF) can be pending
	// at the same time; a bare select picks randomly between ready cases
	// and could drop the tail of the stream. tcpRecv delivers in order on
	// the Poll goroutine, so anything in rxBuf arrived before the error —
	// always drain data first.
	select {
	case data := <-conn.rxBuf:
		n := copy(b, data)
		if n < len(data) {
			conn.pending = data[n:]
		}
		return n, nil
	default:
	}

	select {
	case data := <-conn.rxBuf:
		n := copy(b, data)
		if n < len(data) {
			conn.pending = data[n:]
		}
		return n, nil
	case err := <-conn.rxErr:
		conn.readErr = err
		return 0, err
	}
}

func (conn *TCPConn) Write(b []byte) (int, error) {
	written := 0
	for written < len(b) {
		var wrote int
		var werr error

		// Each round queues as much as currently fits in lwIP's send
		// buffer. The wait for buffer space happens out here, on the
		// caller's goroutine — blocking inside the queued operation would
		// stall the Poll goroutine that services the entire stack.
		queueOperation(func() {
			if conn.closed || conn.pcb == nil {
				werr = errors.New("net: write on closed connection")
				return
			}

			chunk := b[written:]
			if sndbuf := int(conn.pcb.SndBuf()); len(chunk) > sndbuf {
				chunk = chunk[:sndbuf]
			}

			if len(chunk) == 0 {
				// Send buffer is full. Push queued segments toward the
				// wire and let the caller wait for the sent callback.
				if e := conn.pcb.Output(); e != errOk && e != errMem {
					werr = e
				}
				return
			}

			if err := conn.pcb.Write(chunk, tcpWriteFlagCopy); err != nil {
				// ERR_MEM means lwIP could not queue the segment right
				// now; per its contract the caller should wait for
				// tcp_sent and retry. Anything else is fatal.
				if le, ok := err.(lwipError); ok && le == errMem {
					if e := conn.pcb.Output(); e != errOk && e != errMem {
						werr = e
					}
					return
				}
				werr = err
				return
			}
			wrote = len(chunk)

			if e := conn.pcb.Output(); e != errOk && e != errMem {
				werr = e
			}
		})

		if werr != nil {
			return written, werr
		}
		written += wrote

		if wrote == 0 {
			// No progress this round: wait until the stack ACKs in-flight
			// data (tcpSent) or the connection dies (tcpConnErr signals
			// too, and the next round's closed/pcb check reports it).
			<-conn.sendSpace
		}
	}
	return written, nil
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

	// Wake a writer parked waiting for send-buffer space; its next round
	// observes the closed connection and returns an error.
	select {
	case conn.sendSpace <- struct{}{}:
	default:
	}

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
		// Never block the Poll goroutine: rxErr has capacity 1 and Read
		// latches the first terminal error it sees, so a second pending
		// error can be dropped safely.
		select {
		case conn.rxErr <- io.EOF:
		default:
		}
		return errOk
	}

	// Capture the length before freeing the pbuf — reading it afterward is
	// a use-after-free, and a recycled pbuf's length would mis-acknowledge
	// the TCP receive window.
	length := pbuf.TotalLen()

	// Copy out of pbuf into Go-managed memory, then free the pbuf.
	buf := make([]byte, length)
	if length > 0 {
		pbuf.CopyPartial(unsafe.Pointer(&buf[0]), length, 0)
	}
	pbuf.Free()

	// Acknowledge received bytes so lwIP opens the TCP window.
	pcb.Recved(length)

	if length > 0 {
		conn.rxBuf <- buf
	}

	return errOk
}

// tcpSent is invoked by lwIP when the remote peer ACKs previously sent data,
// freeing space in the send buffer. It wakes a writer parked in Write waiting
// for room. The non-blocking send collapses bursts of ACKs into one wakeup.
//
//go:export tcpSent tcp_sent_callback
func tcpSent(arg unsafe.Pointer, pcb *tcpControlBlock, length uint16) lwipError {
	conn := (*TCPConn)(arg)
	select {
	case conn.sendSpace <- struct{}{}:
	default:
	}
	return errOk
}
