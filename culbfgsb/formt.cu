/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#endif

#include "culbfgsb/lbfgsbcuda.h"

namespace lbfgsbcuda {
namespace cuda {
namespace formt {
template <typename real>
__global__ void kernel0(const int col, real* wt, const real* ss,
                        const real theta) {
  const int j = threadIdx.x;
  wt[j] = theta * ss[j];
}

template <typename real>
__global__ void kernel1(const int col, const real* sy, const real* ss, real* wt,
                        const int iPitch, const real theta) {
  const int i = blockIdx.y + 1;
  const int j = blockIdx.x * blockDim.y + threadIdx.y;

  if (j < i || j >= col) return;

  const int k1 = min(i, j);
  const int k = threadIdx.x;

  volatile __shared__ real sdata[4][9];

  real mySum = 0;
  if (k < k1) {
    mySum = sy[i * iPitch + k] * sy[j * iPitch + k] / sy[k * iPitch + k];
  }

  sdata[threadIdx.y][k] = mySum;
  __syncthreads();

  if (k < 4) {
    volatile real* smem = sdata[threadIdx.y] + k;
    *smem = mySum = mySum + smem[4];
    *smem = mySum = mySum + smem[2];
    *smem = mySum = mySum + smem[1];
  }

  if (k == 0) {
    wt[i * iPitch + j] = mySum + theta * ss[i * iPitch + j];
  }
}

template <typename real>
void prog01(const int col, const real* sy, const real* ss, real* wt,
            const int iPitch, const real theta, const cudaStream_t& stream) {
  kernel0<real><<<1, col, 0, stream>>>(col, wt, ss, theta);
  debugSync();
  if (col > 1) {
    kernel1<real><<<dim3(iDivUp(col, 4), col - 1), dim3(8, 4), 0, stream>>>(
        col, sy, ss, wt, iPitch, theta);
  }
}

#define INST_HELPER(real)                                                   \
  template void prog01<real>(const int col, const real* sy, const real* ss, \
                             real* wt, const int iPitch, const real theta,  \
                             const cudaStream_t& stream);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace formt
};  // namespace cuda
};  // namespace lbfgsbcuda