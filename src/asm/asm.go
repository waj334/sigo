package asm

type class interface {
	class()
}

type modifier interface {
	modifier()
}

type constraint interface {
	constraint()
}

type clobber string

func (clobber) modifier() {
}

const (
	// Reserve ensures that the allocated register for this constraint is not shared with other constraints.
	Reserve clobber = "&"
)

// RegisterClass describes a register class constraint.
type RegisterClass string

func (RegisterClass) class() {
	// Dummy func...
}

// Register describes a register constraint.
type Register string

func (Register) constraint() {
	// Dummy func...
}

// Alias declares an alias for a constraint in the assembly code block.
type Alias string

// NOTE: The below are intrinsic functions whose bodies are not actually emitted into the application.

// In specifies that the constraint is an input constraint.
func In(...any) constraint

// Out specifies that the constraint is an output constraint.
func Out(...any) constraint

// InOut specifies that the constraint is both an input and output constraint.
func InOut(...any) constraint

// Clobber ensures that the specified register is not used by any of the constraints.
func Clobber(Register) constraint

// Inline emits an inline assembler expression into the current code block.
func Inline(asm string, constraints ...constraint)
