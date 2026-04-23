package net

import (
	"errors"
	"time"
)

// Addr represents a network end point address.
//
// The two methods [Addr.Network] and [Addr.String] conventionally return strings
// that can be passed as the arguments to [Dial], but the exact form
// and meaning of the strings is up to the implementation.
type Addr interface {
	Network() string // name of the network (for example, "tcp", "udp")
	String() string  // string form of address (for example, "192.0.2.1:25", "[2001:db8::1]:80")
}

// Conn is a generic stream-oriented network connection.
//
// Multiple goroutines may invoke methods on a Conn simultaneously.
type Conn interface {
	// Read reads data from the connection.
	// Read can be made to time out and return an error after a fixed
	// time limit; see SetDeadline and SetReadDeadline.
	Read(b []byte) (n int, err error)

	// Write writes data to the connection.
	// Write can be made to time out and return an error after a fixed
	// time limit; see SetDeadline and SetWriteDeadline.
	Write(b []byte) (n int, err error)

	// Close closes the connection.
	// Any blocked Read or Write operations will be unblocked and return errors.
	// Close may or may not block until any buffered data is sent;
	// for TCP connections see [*TCPConn.SetLinger].
	Close() error

	// LocalAddr returns the local network address, if known.
	LocalAddr() Addr

	// RemoteAddr returns the remote network address, if known.
	RemoteAddr() Addr

	// SetDeadline sets the read and write deadlines associated
	// with the connection. It is equivalent to calling both
	// SetReadDeadline and SetWriteDeadline.
	//
	// A deadline is an absolute time after which I/O operations
	// fail instead of blocking. The deadline applies to all future
	// and pending I/O, not just the immediately following call to
	// Read or Write. After a deadline has been exceeded, the
	// connection can be refreshed by setting a deadline in the future.
	//
	// If the deadline is exceeded a call to Read or Write or to other
	// I/O methods will return an error that wraps os.ErrDeadlineExceeded.
	// This can be tested using errors.Is(err, os.ErrDeadlineExceeded).
	// The error's Timeout method will return true, but note that there
	// are other possible errors for which the Timeout method will
	// return true even if the deadline has not been exceeded.
	//
	// An idle timeout can be implemented by repeatedly extending
	// the deadline after successful Read or Write calls.
	//
	// A zero value for t means I/O operations will not time out.
	SetDeadline(t time.Time) error

	// SetReadDeadline sets the deadline for future Read calls
	// and any currently-blocked Read call.
	// A zero value for t means Read will not time out.
	SetReadDeadline(t time.Time) error

	// SetWriteDeadline sets the deadline for future Write calls
	// and any currently-blocked Write call.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	// A zero value for t means Write will not time out.
	SetWriteDeadline(t time.Time) error
}

// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
	IP   [4]byte
	Port int
}

func (a *TCPAddr) Network() string { return "tcp" }

func (a *TCPAddr) String() string {
	return formatIP(a.IP) + ":" + itoa(a.Port)
}

// ResolveTCPAddr parses an address string into a TCPAddr.
// Only literal IPv4 addresses are supported (no DNS).
func ResolveTCPAddr(network, address string) (*TCPAddr, error) {
	switch network {
	case "tcp", "tcp4":
	default:
		return nil, errors.New("net: unsupported network: " + network)
	}

	host, port, err := splitHostPort(address)
	if err != nil {
		return nil, err
	}

	ip, err := parseIPv4(host)
	if err != nil {
		return nil, err
	}

	return &TCPAddr{IP: ip, Port: port}, nil
}

// splitHostPort splits "host:port" into host and numeric port.
func splitHostPort(address string) (string, int, error) {
	colon := -1
	for i := len(address) - 1; i >= 0; i-- {
		if address[i] == ':' {
			colon = i
			break
		}
	}
	if colon < 0 {
		return "", 0, errors.New("net: missing port in address " + address)
	}

	host := address[:colon]
	portStr := address[colon+1:]
	port, err := atoi(portStr)
	if err != nil || port < 0 || port > 65535 {
		return "", 0, errors.New("net: invalid port " + portStr)
	}
	return host, port, nil
}

// parseIPv4 parses a dotted-decimal IPv4 address.
func parseIPv4(s string) ([4]byte, error) {
	var ip [4]byte
	octet := 0
	idx := 0
	hasDigit := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			octet = octet*10 + int(c-'0')
			if octet > 255 {
				return ip, errors.New("net: invalid IP address " + s)
			}
			hasDigit = true
		} else if c == '.' {
			if !hasDigit || idx >= 3 {
				return ip, errors.New("net: invalid IP address " + s)
			}
			ip[idx] = byte(octet)
			idx++
			octet = 0
			hasDigit = false
		} else {
			return ip, errors.New("net: invalid IP address " + s)
		}
	}
	if !hasDigit || idx != 3 {
		return ip, errors.New("net: invalid IP address " + s)
	}
	ip[3] = byte(octet)
	return ip, nil
}

// UDPAddr represents the address of a UDP end point.
type UDPAddr struct {
	IP   [4]byte
	Port int
}

func (a *UDPAddr) Network() string { return "udp" }

func (a *UDPAddr) String() string {
	return formatIP(a.IP) + ":" + itoa(a.Port)
}

// ResolveUDPAddr parses an address string into a UDPAddr.
// Only literal IPv4 addresses are supported (no DNS).
func ResolveUDPAddr(network, address string) (*UDPAddr, error) {
	switch network {
	case "udp", "udp4":
	default:
		return nil, errors.New("net: unsupported network: " + network)
	}

	host, port, err := splitHostPort(address)
	if err != nil {
		return nil, err
	}

	ip, err := parseIPv4(host)
	if err != nil {
		return nil, err
	}

	return &UDPAddr{IP: ip, Port: port}, nil
}

// formatIP formats an IPv4 address as a dotted-decimal string.
func formatIP(ip [4]byte) string {
	return itoa(int(ip[0])) + "." + itoa(int(ip[1])) + "." + itoa(int(ip[2])) + "." + itoa(int(ip[3]))
}

// itoa converts a non-negative integer to its decimal string representation.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// atoi parses a decimal integer string.
func atoi(s string) (int, error) {
	if len(s) == 0 {
		return 0, errors.New("invalid number")
	}
	n := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, errors.New("invalid number")
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}
