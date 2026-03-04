CC      = g++
CFLAGS  = -O2 -Wall -Wextra -Iinclude -std=c++11
CONV    = src/convolution.cpp

BUILD_DIR = build

UNAME := $(shell uname)

# macOS uses a different way to link OpenCL
ifeq ($(UNAME), Darwin)
    LDFLAGS = -framework OpenCL
else
    LDFLAGS = -lOpenCL
endif

all: $(BUILD_DIR)/convolution_tests

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/convolution_tests: src/test.cpp $(CONV) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
