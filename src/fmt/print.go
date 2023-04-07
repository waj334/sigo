package fmt

func Println(a ...any) (n int, err error) {
	for _, v := range a {
		n += len(v.(string))
	}
	return
}
