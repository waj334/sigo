// RUN: FileCheck %s

package main

func nestedBlocks(a int, b, c, d, e, f int) {
	for {
		a += 1
		for i := 0; i < 0; i++ {
			b += 2
			switch i {
			case 0:
				c += 3
			case 1:
				c += 4
			default:
				c += 5
			}
			d += 6
		}
		e += 7
	}
	f += 8
}

func loopGoto() {
	for {
		goto end
	}
end:
}

func loopBreak() {
	for {
		break
	}
}

func loopContinue() {
	for {
		continue
	}
}

func emptyForLoop() {
	for {
	}
}

func forLoop() {
	for true {
	}
}

func boundedFor() {
	for i := 0; i < 10; i++ {
		if i == 5 {
			continue
		} else if i == 7 {
			break
		}
	}
}

func switchStatement() {
	switch 0 {
	case 0, 1, 3:
		fallthrough
	default:
		fallthrough
	case 4:
		goto label
	case 5:
		break
	case 6:
	}

label:
}

func switchEmptyWithTagAndInt(v int) {
	switch v += 1; v {

	}
}

func switchStatementCompareStruct() {
	type s struct{ i int }
	s0 := s{i: 0}
	s1 := s{i: 1}
	s3 := s{i: 2}
	switch s0 {
	case s0:
	case s1:
	case s3:
	}
}

func switchStatementNoDefault() {
	switch 0 {
	case 0, 1, 3:
	case 4:
	case 5:
	}
}

func switchStatementNoTag() {
	switch {
	case true:
	case false:
	default:
	}
}

func switchStatementInit() {
	switch v := true; v {
	case true:
	case false:
	default:
	}
}

func switchStatementInitNoTag() {
	switch v := true; {
	case v == true:
	case v == false:
	default:
	}
}

func logicalAndIfStatement() {
	if true && true {

	}
}

func logicalOrIfStatement() {
	if true || true {

	}
}

func logicalChainIfStatement() {
	if true && true || false {

	}
}

func logicalChainWithParen() {
	if true && (true || false) {

	}
}

func ifStatement() {
	if true {
	}
}

func ifElseStatement(x int) {
	if true {
		x = 0
	} else if true {
		x = 1
	} else if true {
		x = 2
	} else {
		x = 3
	}
	x = 4
}

func forPanic() bool {
	for true {
		panic("terminator")
	}
	return false
}

func ifPanic() bool {
	if true {
		panic("terminator1")
	} else {
		panic("terminator2")
	}
	return false
}

func switchPanic() bool {
	switch true {
	case true:
		panic("terminator1")
	case false:
		panic("terminator2")
	default:
		panic("terminator3")
	}
	return false
}
