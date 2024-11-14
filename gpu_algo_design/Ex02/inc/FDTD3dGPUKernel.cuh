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

#include <cooperative_groups.h>

#include "FDTD3dGPU.h"

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
// #define RADIUS 4

#ifndef MAX_RADIUS_2D
#define MAX_RADIUS_2D \
  10  // Define the maximum radius to limit shared memory usage
#endif
__constant__ float stencil[MAX_RADIUS_2D + 1];

template <int RADIUS>
__global__ void FiniteDifferencesKernel(float *output, const float *input,
                                        const int dimx, const int dimy,
                                        const int dimz) {
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += RADIUS * stride_y + RADIUS;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[RADIUS];
  float behind[RADIUS];
  float current;

  const int tx = ltidx + RADIUS;
  const int ty = ltidy + RADIUS;

  // Check in bounds
  if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS)) validr = false;

  if ((gtidx >= dimx) || (gtidy >= dimy)) validw = false;

  // Preload the "infront" and "behind" data
  for (int i = RADIUS - 2; i >= 0; i--) {
    if (validr) behind[i] = input[inputIndex];
    inputIndex += stride_z;
  }

  if (validr) current = input[inputIndex];

  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0; i < RADIUS; i++) {
    if (validr) infront[i] = input[inputIndex];
    inputIndex += stride_z;
  }

// Step through the xy-planes
//! Its not clear why 9 was chosen. Might be a  point for future optimization
//! 4 <= dimz <= 376
#pragma unroll 9
  for (int iz = 0; iz < dimz; iz++) {
    // Advance the slice (move the thread-front)
    for (int i = RADIUS - 1; i > 0; i--) behind[i] = behind[i - 1];

    behind[0] = current;
    current = infront[0];

#pragma unroll RADIUS
    for (int i = 0; i < RADIUS - 1; i++) infront[i] = infront[i + 1];

    if (validr) infront[RADIUS - 1] = input[inputIndex];

    inputIndex += stride_z;
    outputIndex += stride_z;
    cg::sync(cta);

    // Note that for the work items on the boundary of the problem, the
    // supplied index when reading the halo (below) may wrap to the
    // previous/next row or even the previous/next xy-plane. This is
    // acceptable since a) we disable the output write for these work
    // items and b) there is at least one xy-plane before/after the
    // current plane, so the access will be within bounds.

    // Update the data slice in the local tile
    // Halo above & below
    if (ltidy < RADIUS) {
      tile[ltidy][tx] = input[outputIndex - RADIUS * stride_y];
      tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
    }

    // Halo left & right
    if (ltidx < RADIUS) {
      tile[ty][ltidx] = input[outputIndex - RADIUS];
      tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
    }

    tile[ty][tx] = current;
    cg::sync(cta);

    // Compute the output value
    float value = stencil[0] * current;
#pragma unroll RADIUS
    for (int i = 1; i <= RADIUS; i++) {
      value +=
          stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                        tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    // Store the output value
    if (validw) output[outputIndex] = value;
  }
}

#ifndef MAX_RADIUS_3D
#define MAX_RADIUS_3D 10
#endif

__constant__ float stencil3d[(2 * MAX_RADIUS_3D + 1) * (2 * MAX_RADIUS_3D + 1) *
                             (2 * MAX_RADIUS_3D + 1)];

template <int RADIUS>
__global__ void FiniteDifferences3DBoxKernel(float *output, const float *input,
                                             const int dimx, const int dimy,
                                             const int dimz) {
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int gtidz = blockIdx.z * blockDim.z + threadIdx.z;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int ltidz = threadIdx.z;

  // Shared memory to cache the output
  __shared__ float cache[k_blockDimMaxZ][k_blockDimMaxY][k_blockDimX];

  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  const int stride_y = dimx;
  const int stride_z = stride_y * dimy;

  // Calculate input/output indices
  int inputIndex = (gtidz * stride_z) + (gtidy * stride_y) + gtidx;
  int outputIndex = (gtidz * stride_z) + (gtidy * stride_y) + gtidx;

  // Bounds checking
  if ((gtidx >= dimx) || (gtidy >= dimy) || (gtidz >= dimz)) validw = false;

  // Compute stencil directly from global memory without caching input
  if (validw) {
    float value = 0.0f;
    int stencil_idx = 0;

    // Full 3D box stencil computation
    // unrolling here seems to hurt performance at higher radiuses with lwo to
    // no gains at lower
    for (int z = -RADIUS; z <= RADIUS; z++) {
      for (int y = -RADIUS; y <= RADIUS; y++) {
        for (int x = -RADIUS; x <= RADIUS; x++) {
          int global_x = gtidx + x;
          int global_y = gtidy + y;
          int global_z = gtidz + z;

          // Clamp to boundary
          global_x = max(0, min(dimx - 1, global_x));
          global_y = max(0, min(dimy - 1, global_y));
          global_z = max(0, min(dimz - 1, global_z));

          value += stencil3d[stencil_idx] * input[inputIndex];
          stencil_idx++;
        }
      }
    }

    // Cache the output value in shared memory
    cache[ltidz][ltidy][ltidx] = value;
  }

  cg::sync(cta);

  // Write the cached output to global memory
  if (validw) {
    output[outputIndex] = cache[ltidz][ltidy][ltidx];
  }
}

__global__ void emptyKernel() {}
