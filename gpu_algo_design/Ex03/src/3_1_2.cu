#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void __lane_min_int(curandState *state) {
  int idx = threadIdx.x;

  // Initialize cuRAND
  curand_init(1234, idx, 0, &state[idx]);

  // Generate a random positive integer
  int local_value = static_cast<int>(curand_uniform(&state[idx]) *
                                     1000);  // Random positive integer
  printf("Thread %d has value %d\n", idx, local_value);

  int min_value = local_value;
  int min_index = idx;

  // Use __reduce_min_sync to find the minimum value in the warp
  // Disabled Code
  //int warp_min_value = __reduce_min_sync(0xffffffff, local_value);
  //int warp_min_index = __reduce_min_sync(0xffffffff, idx);

  // // shuffle down to find the min value and its index
  // for (int i = 16; i > 0; i /= 2) {
  //   //int temp_value = __shfl_down_sync(0xffffffff, min_value, i);
  //   //int temp_index = __shfl_down_sync(0xffffffff, min_index, i);
  //   if (temp_value < min_value) {
  //     min_value = temp_value;
  //     min_index = temp_index;
  //   }
  // }
  int rotated_value = __shfl_sync(0xFFFFFFFF, local_value, (threadIdx.x -1 ) & (32 - 1));

  min_index = __shfl_sync(0xffffffff, min_index, 0);
  printf("Thread %d, New Value %d Old Value %d\n",idx, rotated_value, local_value);

}

int main() {
  printf("test\n");
  // Allocate space for cuRAND state
  curandState *d_state;
  cudaMalloc(&d_state, 32 * sizeof(curandState));

  // Launch kernel
  __lane_min_int<<<1, 32>>>(d_state);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // Free cuRAND state
  cudaFree(d_state);

  return 0;
}