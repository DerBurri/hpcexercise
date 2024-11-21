/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// GLobal accummulator value
__device__ float globalAcc = 0;

bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
      (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    atomicAdd(&globalAcc, mySum);
    //printf("Block %d, Thread %d, sum: %f, globAcc %f\n", blockIdx.x, threadIdx.x, mySum, globalAcc);
  }
}

template <class T>
void reduceSpeciale(int size, int threads, int blocks, T *d_idata, T *d_odata, int dev_num) {

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // For reduce7 kernel we require only blockSize/warpSize
  // number of elements in shared memory
  int smemSize = ((threads / 32) + 1) * sizeof(T);

  void* args[3] = {d_idata, d_odata, &size};

#define STANDARD_KERNEL_LAUNCH(threads, isPowOf2) case threads: reduce7<T, threads, isPowOf2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size); break;

#define COOP_KERNEL_LAUNCH(threads, isPowOf2) case threads: cudaLaunchCooperativeKernel((void*)reduce7<T, threads, isPowOf2>, dimGrid, dimBlock, args, smemSize, 0); break;
  
  if (isPow2(size)) {
    switch (threads) {
      STANDARD_KERNEL_LAUNCH(1024,true)
      STANDARD_KERNEL_LAUNCH(512,true)
      STANDARD_KERNEL_LAUNCH(256,true)
      STANDARD_KERNEL_LAUNCH(128,true)
      STANDARD_KERNEL_LAUNCH(64,true)
      STANDARD_KERNEL_LAUNCH(32,true)
      STANDARD_KERNEL_LAUNCH(16,true)
      STANDARD_KERNEL_LAUNCH(8,true)
      STANDARD_KERNEL_LAUNCH(4,true)
      STANDARD_KERNEL_LAUNCH(2,true)
      STANDARD_KERNEL_LAUNCH(1,true)
    }
  } else {
    switch (threads) {
      STANDARD_KERNEL_LAUNCH(1024,true)
      STANDARD_KERNEL_LAUNCH(512,false)
      STANDARD_KERNEL_LAUNCH(256,false)
      STANDARD_KERNEL_LAUNCH(128,false)
      STANDARD_KERNEL_LAUNCH(64,false)
      STANDARD_KERNEL_LAUNCH(32,false)
      STANDARD_KERNEL_LAUNCH(16,false)
      STANDARD_KERNEL_LAUNCH(8,false)
      STANDARD_KERNEL_LAUNCH(4,false)
      STANDARD_KERNEL_LAUNCH(2,false)
      STANDARD_KERNEL_LAUNCH(1,false)
    }
  }
}

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
__device__ void reduceBlock(volatile float *sdata, float mySum,
                            const unsigned int tid, cg::thread_block cta) {
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  sdata[tid] = mySum;
  cg::sync(tile32);

  const int VEC = 32;
  const int vid = tid & (VEC - 1);

  float beta = mySum;
  float temp;

  for (int i = VEC / 2; i > 0; i >>= 1) {
    if (vid < i) {
      temp = sdata[tid + i];
      beta += temp;
      sdata[tid] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0) {
    beta = 0;
    for (int i = 0; i < blockDim.x; i += VEC) {
      beta += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocks(const float *g_idata, float *g_odata,
                             unsigned int n, cg::thread_block cta) {
  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // do reduction in shared mem
  reduceBlock<blockSize>(sdata, mySum, tid, cta);

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceMultiPass(const float *g_idata, float *g_odata,
                                unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
}

// Global variable used by reduceSinglePass to count how many blocks have
// finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt) {
  return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0,
                            cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation It does so by keeping track of how many blocks have finished.
// After each thread block completes the reduction of its own block of data, it
// "takes a ticket" by atomically incrementing a global counter.  If the ticket
// value is equal to the number of thread blocks, then the block holding the
// ticket knows that it is the last block to finish.  This last block is
// responsible for summing the results of all the other blocks.
//
// In order for this to work, we must be sure that before a block takes a
// ticket, all of its memory transactions have completed.  This is what
// __threadfence() does -- it blocks until the results of all outstanding memory
// transactions within the calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata,
                                 unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  //
  // PHASE 1: Process all inputs assigned to this block
  //

  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);

  //
  // PHASE 2: Last block finished will process all partial sums
  //

  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    __shared__ bool amLast;
    extern float __shared__ smem[];

    // wait until all outstanding memory instructions in this thread are
    // finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == gridDim.x - 1);
    }

    cg::sync(cta);

    // The last block sums the results of all other blocks
    if (amLast) {
      int i = tid;
      float mySum = 0;

      while (i < gridDim.x) {
        mySum += g_odata[i];
        i += blockSize;
      }

      reduceBlock<blockSize>(smem, mySum, tid, cta);

      if (tid == 0) {
        g_odata[0] = smem[0];

        // reset retirement count so that next run succeeds
        retirementCount = 0;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduce(int size, int threads, int blocks, float *d_idata,
                       float *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
      case 512:
        reduceMultiPass<512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduceMultiPass<256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduceMultiPass<128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduceMultiPass<64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduceMultiPass<32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduceMultiPass<16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduceMultiPass<8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduceMultiPass<4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduceMultiPass<2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduceMultiPass<1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceMultiPass<512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduceMultiPass<256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduceMultiPass<128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduceMultiPass<64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduceMultiPass<32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduceMultiPass<16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduceMultiPass<8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduceMultiPass<4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduceMultiPass<2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduceMultiPass<1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

extern "C" void reduceSinglePass(int size, int threads, int blocks,
                                 float *d_idata, float *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
      case 512:
        reduceSinglePass<512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduceSinglePass<256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduceSinglePass<128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduceSinglePass<64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduceSinglePass<32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduceSinglePass<16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduceSinglePass<8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduceSinglePass<4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduceSinglePass<2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduceSinglePass<1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceSinglePass<512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduceSinglePass<256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduceSinglePass<128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduceSinglePass<64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduceSinglePass<32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduceSinglePass<16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduceSinglePass<8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduceSinglePass<4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduceSinglePass<2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduceSinglePass<1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

#endif  // #ifndef _REDUCE_KERNEL_H_
