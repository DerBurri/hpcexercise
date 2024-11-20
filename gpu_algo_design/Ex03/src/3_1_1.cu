#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define SHFL_DELTA 4
__global__ void __shfl_up_wrap(curandState *state) {
  int idx = threadIdx.x;

  // Initialize cuRAND
  curand_init(1234, idx, 0, &state[idx]);

  // Generate a random positive integer
  int local_value = static_cast<int>(curand_uniform(&state[idx]) *
                                     1000);  // Random positive integer

  int rotated_value = __shfl_sync(0xFFFFFFFF, local_value, (threadIdx.x - SHFL_DELTA ) & (32 - 1));
  printf("Thread %d, New Value %d Old Value %d\n",idx, rotated_value, local_value);

}

int main() {
  printf("test\n");
  // Allocate space for cuRAND state
  curandState *d_state;
  cudaMalloc(&d_state, 32 * sizeof(curandState));

  // Launch kernel
  __shfl_up_wrap<<<1, 32>>>(d_state);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Free cuRAND state
  cudaFree(d_state);

  return 0;
}