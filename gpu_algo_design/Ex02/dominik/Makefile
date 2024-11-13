
# Compiler
NVCC = nvcc

OUT = ./bin/FDTD3d
BIN = ./bin/

# Default compilation flags
CFLAGS = -g -G -ccbin g++ -m64 -diag-suppress=177

# Include directory
INCLUDE_DIRS = /opt/asc/spack/opt/spack/linux-debian12-x86_64_v2/gcc-13.1.0/cuda-12.6.1-eq5axkonpabssoacghdsg3q4nas5nkja/include
INCLUDE_DIRS += ./inc

# CUDA path
CUDA_PATH = /opt/asc/spack/opt/spack/linux-debian12-x86_64_v2/gcc-13.1.0/cuda-12.6.1-eq5axkonpabssoacghdsg3q4nas5nkja

# Compiler flags
CFLAGS = -g -G -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -diag-suppress=177


# GPU-specific compilation flags
CFLAGS_4080 = -ccbin g++ -m64 -gencode arch=compute_89,code=sm_89 -diag-suppress=177
CFLAGS_2080ti = -ccbin g++ -m64 -gencode arch=compute_75,code=sm_75 -diag-suppress=177

# Default target
all: clean build

# Build targets
build: $(OUT)

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
$(OUT): $(BIN)FDTD3d.o $(BIN)FDTD3dGPU.o $(BIN)FDTD3dReference.o
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ $+

$(BIN)FDTD3d.o: src/FDTD3d.cpp
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ -c $<

$(BIN)FDTD3dGPU.o: src/FDTD3dGPU.cu
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ -c $<

$(BIN)FDTD3dReference.o: src/FDTD3dReference.cpp
	$(NVCC) -t 4 $(CFLAGS) $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) -o $@ -c $<

# Clean rules
clean:
	rm -rf ./bin/*
	rm -rf ./src/*.o

clobber: clean