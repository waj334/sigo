package main

import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
	Use:   "sigo",
	Short: "Sigo is an implementation of the Go language compiler for embedded systems!",
	Long: `Sigo is an implementation of the Go language compiler and Go runtime for embedded systems based on the AVR, 
				ARM Cortex-M, Xtensa or RISCV MCU architectures!`,
}

func init() {
	rootCmd.AddCommand(buildCmd)
}

func main() {
	rootCmd.Execute()
}
