package builder

import "errors"

var (
	ErrParserError          = errors.New("parser error occurred")
	ErrMultiplePackages     = errors.New("directory contained multiple packages")
	ErrUnexpectedOutputPath = errors.New("unexpected output path provided")
	ErrLinknameAlreadyUsed  = errors.New("encountered duplicate linkname")
)
