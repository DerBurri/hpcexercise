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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "histogram_common.h"

////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////

template <uint binNum, uint binNumLog2>
inline __device__ void addWord(uint *s_WarpHist, uint data) {

  /*
     This computation might look overtly complicatied but it basically says 
     that only the top bits of the input number (parameter data) are used to 
     index a bin in a histogram. For example, if the datatype of the
     input array (here uint) is 32 bits wide and we have 1024 bins, these bins
     can be indexed by a 10 bit number (range 0 - 1023). This means that we 
     have to split the range of the input number (0 - 2**23) into 1024 equally 
     long ranges. And this can be done by choosing only the top 10 bits of the
     inptu number.
  */
  uint binIdx = (data >> (sizeof(uint)*8 - binNumLog2)) & (binNum -1);
  atomicAdd(s_WarpHist + binIdx, 1);
}

template <uint binNum, uint binNumLog2>
__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount, uint warpCount) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // Per-block (!!!) subhistogram storage which is shared among multiple
  // warps.
  extern __shared__ uint s_Hist[];

  // Clear shared memory storage for current threadblock before processing
#pragma unroll
  for (uint i = threadIdx.x; i < binNum; i+= warpCount * WARP_SIZE) 
    s_Hist[i] = 0;

  cg::sync(cta);

  // Cycle through the entire data set, update subhistograms for each warp.
  // "pos" variable starts at a global index of a thread and strides with the 
  // count of all launched threads (since there can be less threads than 
  // allocated data)
  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount;
       pos += UMUL(blockDim.x, gridDim.x)) {

    uint data = d_Data[pos];
    addWord<binNum, binNumLog2>(s_Hist, data);
  }

  cg::sync(cta);

  // Stride through the shared memory to assign the per-block subhistogram into 
  // global memory.
  for (uint bin = threadIdx.x; bin < binNum; bin += (warpCount * WARP_SIZE)) {
    d_PartialHistograms[blockIdx.x * binNum + bin] = s_Hist[bin];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256Kernel() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram256Kernel(uint *d_Histogram,
                                        uint *d_PartialHistograms,
                                        uint histogramCount, 
                                        uint binNum) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  uint sum = 0;

  for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
    sum += d_PartialHistograms[blockIdx.x + i * binNum];
  }

  __shared__ uint data[MERGE_THREADBLOCK_SIZE];
  data[threadIdx.x] = sum;

  for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);

    if (threadIdx.x < stride) {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    d_Histogram[blockIdx.x] = data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
// histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 224;
static uint *d_PartialHistograms;

// Internal memory allocation
extern "C" void initHistogram256(uint binNum) {
  checkCudaErrors(cudaMalloc(
      (void **)&d_PartialHistograms,
      PARTIAL_HISTOGRAM256_COUNT * binNum * sizeof(uint)));
}

// Internal memory deallocation
extern "C" void closeHistogram256(void) {
  checkCudaErrors(cudaFree(d_PartialHistograms));
}

extern "C" void histogram256(uint *d_Histogram, void *d_Data, uint byteCount, uint binNum, uint warpCount) {
  assert(byteCount % sizeof(uint) == 0);
#ifdef DEBUG
  printf("histogram: Launch config: %d, %d, %d\n", PARTIAL_HISTOGRAM256_COUNT, 
      warpCount * WARP_SIZE, warpCount * binNum * sizeof(uint));
#endif

#define KERNEL_CALL(binNum, binNumLog2) \
  case binNum: histogram256Kernel<binNum, binNumLog2><<<PARTIAL_HISTOGRAM256_COUNT, \
  warpCount * WARP_SIZE, \
  binNum * sizeof(uint)>>>( \
  d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint), warpCount); \
  break;

  switch (binNum) {
    KERNEL_CALL(256,8)
    KERNEL_CALL(512,9)
    KERNEL_CALL(1024,10)
    KERNEL_CALL(2048,11)
    KERNEL_CALL(4096,12)
    KERNEL_CALL(8192,13)
  }

  getLastCudaError("histogram256Kernel() execution failed\n");

#ifdef DEBUG
  printf("mergeHistogram: Launch config: %d, %d\n", binNum, MERGE_THREADBLOCK_SIZE);
#endif

  mergeHistogram256Kernel<<<binNum, MERGE_THREADBLOCK_SIZE>>>(
      d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT, binNum);
  getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}
