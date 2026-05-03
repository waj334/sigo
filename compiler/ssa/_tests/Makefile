SIGOC ?= sigoc
CPU ?= stm32h747xi
FLOAT ?= hardfp
STACK_SIZE ?= 8192
OPT ?= -O0
DBG ?= -g
LD ?= ld.lld

# Each subdirectory containing a main.go is treated as a test program.
TESTS := $(patsubst %/main.go,%,$(wildcard */main.go))

BIN := bin

ALL_BINS := $(addprefix $(BIN)/,$(addsuffix .elf,$(TESTS)))

COMMON_SRCS := $(wildcard common/*.go)

.PHONY: all clean list $(TESTS)

all: $(ALL_BINS)

list:
	@echo "Tests:"
	@for t in $(TESTS); do echo "  $$t"; done

# Build rule: bin/<name>.elf from <name>/main.go and shared common sources.
# Each test rebuilds unconditionally; sigoc handles incremental work itself.
$(BIN)/%.elf: %/main.go $(COMMON_SRCS) | $(BIN)
	LD=$(LD) $(SIGOC) build --cpu $(CPU) $(DBG) $(OPT) --float $(FLOAT) --stack-size=$(STACK_SIZE) -o $@ ./$*

# Phony per-test target so `make rangefunc_basic` works.
$(TESTS): %: $(BIN)/%.elf

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
