package http

type commonError string

func (e commonError) Error() string {
	return string(e)
}

const (
	ErrHeaderFull        commonError = "http: header table full"
	ErrTooManyHeaders    commonError = "http: too many headers in response"
	ErrHeadersTooLarge   commonError = "http: response headers exceed buffer size"
	ErrMalformedResponse commonError = "http: malformed response"
	ErrBadContentLength  commonError = "http: invalid Content-Length"
	ErrBadChunkSize      commonError = "http: invalid chunk size"
	ErrBufferTooSmall    commonError = "http: response buffer too small"
	ErrBodyClosed        commonError = "http: body already closed"
	ErrUnexpectedEOF     commonError = "http: unexpected EOF in response body"
	ErrChunkedAfterRead  commonError = "http: malformed chunked encoding"
	ErrBodyState         commonError = "http: corrupt body state"
)
