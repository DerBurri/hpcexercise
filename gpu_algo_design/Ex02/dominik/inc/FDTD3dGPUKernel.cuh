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
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cooperative_groups.h>

#include "FDTD3dGPU.h"

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
// #define RADIUS 4

#define MAX_RADIUS_2D \
  10  // Define the maximum radius to limit shared memory usage
#define MAX_RADIUS_3D 10
__constant__ float stencil[MAX_RADIUS_2D + 1];  // Maximum RADIUS is 10
__constant__ float stencil3D[(2 * MAX_RADIUS_3D + 1) * (2 * MAX_RADIUS_3D + 1) *
                             (2 * MAX_RADIUS_3D + 1)];

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
//! TODO look at this loop unroll, its not clear why 9 was chosen. Might be a
//! point for future optimization 4 <= dimz <= 376
#pragma unroll 9
  for (int iz = 0; iz < dimz; iz++) {
    // Advance the slice (move the thread-front)
    for (int i = RADIUS - 1; i > 0; i--) {
      behind[i] = behind[i - 1];
    }
    behind[0] = current;
    current = infront[0];
#pragma unroll RADIUS
    for (int i = 0; i < RADIUS - 1; i++) {
      infront[i] = infront[i + 1];
    }

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

//! 3D
template <int RADIUS>
__global__ void FullStencilKernel(float *output, const float *input,
                                  const int dimx, const int dimy,
                                  const int dimz) {
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int gtidz = blockIdx.z * blockDim.z + threadIdx.z;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int ltidz = threadIdx.z;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  const int workz = blockDim.z;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxZ + 2 * RADIUS]
                       [k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

  float current = 0;

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += RADIUS + RADIUS * stride_y + RADIUS * stride_z;
  // Advance inputIndex to target element
  inputIndex += gtidx + gtidy * stride_y + gtidz * stride_z;
  outputIndex = inputIndex;

  const int tx = ltidx + RADIUS;
  const int ty = ltidy + RADIUS;
  const int tz = ltidz + RADIUS;

  // Check in bounds
  bool validr = true;
  bool validw = true;
  if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS) ||
      (gtidz >= dimz + RADIUS))
    validr = false;
  if ((gtidx >= dimx) || (gtidy >= dimy) || (gtidz >= dimz)) validw = false;

  // HALO z
  if (ltidz < RADIUS) {
    tile[ltidz][ty][tx] = input[inputIndex - RADIUS * stride_z];
    tile[ltidz + workz + RADIUS][ty][tx] = input[inputIndex + workz * stride_z];
  }
  // Halo y
  if (ltidy < RADIUS) {
    tile[tz][ltidy][tx] = input[inputIndex - RADIUS * stride_y];
    tile[tz][ltidy + worky + RADIUS][tx] = input[inputIndex + worky * stride_y];
  }
  // Halo x
  if (ltidx < RADIUS) {
    tile[tz][ty][ltidx] = input[inputIndex - RADIUS];
    tile[tz][ty][ltidx + workx + RADIUS] = input[inputIndex + workx];
  }

  tile[tz][ty][tx] = current;
  cg::sync(cta);

  // Compute the output value
  float value = stencil3D[0] * current;
  // 3D convolution computation
  for (int i = 0; i < RADIUS; i++) {
    for (int j = 0; j < RADIUS; j++) {
      for (int k = 0; k < RADIUS; k++) {
        // value += stencil3D[(i + 1) * (2 * RADIUS + 1) * (2 * RADIUS + 1) +
        //  (j + 1) * (2 * RADIUS + 1) + (k + 1)] *
        value += stencil3D[(i + 1) * (2 * RADIUS) * (2 * RADIUS) +
                           (j + 1) * (2 * RADIUS) + (k + 1)] *
                 (tile[tz - i][ty - j][tx - k] + tile[tz - i][ty - j][tx + k] +
                  tile[tz - i][ty + j][tx - k] + tile[tz - i][ty + j][tx + k] +
                  tile[tz + i][ty - j][tx - k] + tile[tz + i][ty - j][tx + k] +
                  tile[tz + i][ty + j][tx - k] + tile[tz + i][ty + j][tx + k]);
      }
    }
  }

  // Store the output value
  if (validw) output[outputIndex] = value;
}

// Instantiate kernels for radii 1 to 10
template __global__ void FiniteDifferencesKernel<1>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<2>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<3>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<4>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<5>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<6>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<7>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<8>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<9>(float *output,
                                                    const float *input,
                                                    const int dimx,
                                                    const int dimy,
                                                    const int dimz);
template __global__ void FiniteDifferencesKernel<10>(float *output,
                                                     const float *input,
                                                     const int dimx,
                                                     const int dimy,
                                                     const int dimz);
template __global__ void FullStencilKernel<1>(float *output, const float *input,
                                              const int dimx, const int dimy,
                                              const int dimz);
template __global__ void FullStencilKernel<2>(float *output, const float *input,
                                              const int dimx, const int dimy,
                                              const int dimz);
template __global__ void FullStencilKernel<3>(float *output, const float *input,
                                              const int dimx, const int dimy,
                                              const int dimz);
template __global__ void FullStencilKernel<4>(float *output, const float *input,
                                              const int dimx, const int dimy,
                                              const int dimz);