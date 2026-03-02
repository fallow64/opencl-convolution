CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
TARGET  = opencl-convolution2d

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    LDFLAGS = -framework OpenCL
else
    LDFLAGS = -lOpenCL
endif

SOURCES = $(wildcard *.c)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
