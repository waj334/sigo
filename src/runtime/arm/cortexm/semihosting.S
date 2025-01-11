.global runtime_arm_cortexm.readSemihosting
.type runtime_arm_cortexm.readSemihosting, %function
runtime_arm_cortexm.readSemihosting:
	mov r0, #0x07
	bkpt 0xAB
	bx lr
.size runtime_arm_cortexm.readSemihosting, .-runtime_arm_cortexm.readSemihosting

.global runtime_arm_cortexm.writeSemihosting
.type runtime_arm_cortexm.writeSemihosting, %function
runtime_arm_cortexm.writeSemihosting:
  mov r1, r0
	mov r0, #0x03
	bkpt 0xAB
	bx lr
.size runtime_arm_cortexm.writeSemihosting, .-runtime_arm_cortexm.writeSemihosting