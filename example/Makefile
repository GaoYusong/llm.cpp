CC = clang
CFLAGS = -O3 -Ofast -std=c++20 -g
LDFLAGS =
LDLIBS = -lm
INCLUDES =

# Detect the C++ standard library to link against
ifeq ($(shell echo 'int main(){}' | $(CC) -lc++ -x c++ - -o /dev/null 2>/dev/null; echo $$?), 0)
  LDLIBS += -lc++
else
  LDLIBS += -lstdc++
endif

# Check if OpenMP is available
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
ifeq ($(shell printf '\#include <omp.h>\nint main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null > /dev/null 2>&1; echo $$?), 0)
  ifeq ($(shell uname), Darwin)
    # macOS with Homebrew
    CFLAGS += -Xclang -fopenmp
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib
    LDLIBS += -lomp
    INCLUDES += -I/opt/homebrew/opt/libomp/include
  else
    # Ubuntu or other Linux distributions
    CFLAGS += -fopenmp
    LDLIBS += -lgomp
  endif
  $(info NICE Compiling with OpenMP support)
else
  $(warning OOPS Compiling without OpenMP support)
endif

# PHONY means these targets will always be executed
.PHONY: all example clean

# default target is all
all: example

example: tinytorch_example.cpp
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f example
