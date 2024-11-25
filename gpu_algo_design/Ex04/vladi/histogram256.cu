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

#define TAG_MASK 0xFFFFFFFFU
inline __device__ void addByte(uint *s_WarpHist, uint data) {
  // Increments a bin of a value "data" by 1
  atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU);
  addByte(s_WarpHist, (data >> 8) & 0xFFU);
  addByte(s_WarpHist, (data >> 16) & 0xFFU);
  addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount, uint binNum, uint warpCount) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // Per-warp subhistogram storage
  extern __shared__ uint s_Hist[];
  uint *s_WarpHist = s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * binNum;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

  for (uint i = 0;
       i < (binNum / WARP_SIZE);
       i++) {
    s_Hist[threadIdx.x + i * (warpCount * WARP_SIZE)] = 0;
  }

  cg::sync(cta);

  // Cycle through the entire data set, update subhistograms for each warp
  // pos starts at a global index of a thread
  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount;
       // Stride with the count of all launched threads (since there can be less threads
       // Htan allocated data
       pos += UMUL(blockDim.x, gridDim.x)) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data);
  }

  // Merge per-warp histograms into per-block and write to global memory
  cg::sync(cta);

  for (uint bin = threadIdx.x; bin < binNum;
       bin += (warpCount * WARP_SIZE)) {
    uint sum = 0;

    for (uint i = 0; i < warpCount; i++) {
      sum += s_Hist[bin + i * binNum] & TAG_MASK;
    }

    d_PartialHistograms[blockIdx.x * binNum + bin] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
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

  histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT,
                       warpCount * WARP_SIZE,
                       warpCount * binNum * sizeof(uint)>>>(
      d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint), binNum, warpCount);
  getLastCudaError("histogram256Kernel() execution failed\n");

#ifdef DEBUG
  printf("mergeHistogram: Launch config: %d, %d\n", binNum, MERGE_THREADBLOCK_SIZE);
#endif

  mergeHistogram256Kernel<<<binNum, MERGE_THREADBLOCK_SIZE>>>(
      d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT, binNum);
  getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}
