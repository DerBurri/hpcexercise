
# Compiler
NVCC = $(CUDA_HOME)/bin/nvcc

OUT = ./bin/min
BIN = ./bin/
SRC = ./

# Default compilation flags
CFLAGS = -g -G -ccbin g++ -m64 -diag-suppress=177

# Include directory
INCLUDE_DIRS = $(CUDA_HOME)/include
INCLUDE_DIRS += ./inc

# Compiler flags
CFLAGS = -g -G -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -diag-suppress=177


# GPU-specific compilation flags
CFLAGS_4080 = -ccbin g++ -m64 -gencode arch=compute_89,code=sm_89 -diag-suppress=177
CFLAGS_2080ti = -ccbin g++ -m64 -gencode arch=compute_75,code=sm_75 -diag-suppress=177

# Default target
all: clean build

# Build targets
3_1_2: CFLAGS := -ccbin g++ -m64  -gencode arch=compute_89,code=sm_89
3_1_2: clean
3_1_2: ./bin/3_1_2

3_1_1: CFLAGS := -ccbin g++ -m64  -gencode arch=compute_89,code=sm_89
3_1_1: clean
3_1_1: ./bin/3_1_1

build: 3_1_1 3_1_2

debug: CFLAGS += -DGPU_PROFILING=1 -DDEBUG=1
debug: clean build

prof: CFLAGS += -DGPU_PROFILING=1 -DSUPRESS_OUTPUT=1
prof: clean build

# GPU-specific targets
4080: CFLAGS := $(CFLAGS_4080)
4080: clean
4080: build

2080ti: CFLAGS := $(CFLAGS_2080ti)
2080ti: clean
2080ti: build

4080_debug: CFLAGS := $(CFLAGS_4080) -g -G
4080_debug: clean
4080_debug: debug

2080ti_debug: CFLAGS := $(CFLAGS_2080ti) -g -G 
2080ti_debug: clean
2080ti_debug: debug

4080_prof: CFLAGS := $(CFLAGS_4080)
4080_prof: clean
4080_prof: prof

2080ti_prof: CFLAGS := $(CFLAGS_2080ti)
2080ti_prof: clean
2080ti_prof: prof

# Build rules
bin/3_1_2: $(SRC)3_1_2.cu
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ $<

bin/3_1_1: $(SRC)3_1_1.cu
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ $<

# Clean rules
clean:
	rm -rf ./bin/*
	rm -rf ./*.o

clobber: clean
