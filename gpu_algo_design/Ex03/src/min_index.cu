#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void __warp_min_idx(curandState *state) {
  int idx_min;
  int idx = threadIdx.x;
  int min_value;
  int min_index = idx;

  // Initialize cuRAND
  int seed = 1234;
  curand_init(seed, idx, 0, &state[idx]);

  // Generate a random positive integer
  int local_value = static_cast<int>(curand_uniform(&state[idx]) *
                                     1000);  // Random positive integer
  printf("Thread %d has value %d\n", idx, local_value);
  min_value = local_value;

  // shuffle down to find the min value and its index
  for (int i = 16; i > 0; i /= 2) {
    int temp_value = __shfl_down_sync(0xffffffff, min_value, i);
    int temp_index = __shfl_down_sync(0xffffffff, min_index, i);
    if (temp_value < min_value) {
      min_value = temp_value;
      min_index = temp_index;
    }
  }

  min_index = __shfl_sync(0xffffffff, min_index, 0);
  if (threadIdx.x == 0) {
    printf("Lane index of the min value in warp is %d\n",
           idx_min);
    printf("Min value in warp is %d\n", min_value);
  }
}

int main() {
  printf("test\n");
  // Allocate space for cuRAND state
  curandState *d_state;
  cudaMalloc(&d_state, 32 * sizeof(curandState));

  // Launch kernel
  __warp_min_idx<<<1, 32>>>(d_state);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Free cuRAND state
  cudaFree(d_state);

  return 0;
}