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
namespace cmprlb {
template <typename real>
__global__ void kernel0(int n, real* r, const real* g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  r[i] = -g[i];
}

template <int bsize, typename real>
__global__ void kernel1(int nfree, const int* index, const int col,
                        const int head, const int m, const int iPitch,
                        const real* wa, const real* wy, const real* ws,
                        const real theta, const real* z, const real* x,
                        const real* g, real* r) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  const int tidx = threadIdx.x;  // 8
  const int tidy = threadIdx.y;  // 64

  volatile __shared__ real sdata[(512 / bsize)][bsize + 1];

  __shared__ real a[2][bsize + 1];

  real mySum;

  if (tidy == 0 && tidx < col) {
    a[0][tidx] = wa[tidx];
    a[1][tidx] = theta * wa[col + tidx];
  }
  int k = 0;
  if (i < nfree && tidx < col) {
    const int pointr = Modular((head + tidx), m);
    k = index[i];
    __syncthreads();

    mySum = wy[k * iPitch + pointr] * a[0][tidx] +
            ws[k * iPitch + pointr] * a[1][tidx];
  } else
    mySum = 0;

  if (bsize > 1) {
    volatile real* smem = sdata[tidy] + tidx;
    *smem = mySum;

    __syncthreads();

    if (bsize > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bsize > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bsize > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tidx == 0 && i < nfree) {
    r[i] = -theta * (z[k] - x[k]) - g[k] + mySum;
  }
}

template <typename real>
void prog0(const int n, real* r, const real* g, const cudaStream_t& stream) {
  kernel0<real><<<dim3(iDivUp(n, 512)), dim3(512), 0, stream>>>(n, r, g);
}

template <typename real>
void prog1(const int nfree, const int* index, const int col, const int head,
           const int m, const int iPitch, const real* wa, const real* wy,
           const real* ws, const real theta, const real* z, const real* x,
           const real* g, real* r, const cudaStream_t& stream) {
  if (col > 4) {
    int nblocky = 512 / 8;
    kernel1<8, real><<<dim3(iDivUp(nfree, nblocky)), dim3(8, nblocky), 0, stream>>>(
        nfree, index, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
  } else if (col > 2) {
    int nblocky = 512 / 4;
    kernel1<4, real>
        <<<dim3(iDivUp(nfree, nblocky)), dim3(4, nblocky), 0, stream>>>(
        nfree, index, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
  } else if (col > 1) {
    int nblocky = 512 / 2;
    kernel1<2, real>
        <<<dim3(iDivUp(nfree, nblocky)), dim3(2, nblocky), 0, stream>>>(
        nfree, index, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
  } else if (col == 1) {
    int nblocky = 512 / 1;
    kernel1<1, real>
        <<<dim3(iDivUp(nfree, nblocky)), dim3(1, nblocky), 0, stream>>>(
        nfree, index, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
  }
}

#define INST_HELPER(real)                                                      \
  template void prog0<real>(const int, real*, const real*,                     \
                            const cudaStream_t&);                              \
  template void prog1<real>(const int, const int*, const int, const int,       \
                            const int, const int, const real*, const real*,    \
                            const real*, const real, const real*, const real*, \
                            const real*, real*, const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace cmprlb
};  // namespace cuda
};  // namespace lbfgsbcuda