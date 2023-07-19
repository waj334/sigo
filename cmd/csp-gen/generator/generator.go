package generator

type Generator interface {
	Generate(out string) error
}
