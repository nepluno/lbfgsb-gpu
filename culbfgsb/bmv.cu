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

#include <cublas_v2.h>

#include "culbfgsb/lbfgsbcuda.h"

namespace lbfgsbcuda {
namespace cuda {
namespace bmv {
template <typename real>
__global__ void kernel0(const real* sy, const int col, const real* v,
                        const int iPitch, real* p) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  const int k = threadIdx.x;
  const int i2 = col + i;

  volatile __shared__ real sdata[4][9];

  real mySum = 0;
  if (k < i && i < col) {
    mySum = sy[i * iPitch + k] * v[k] / sy[k * iPitch + k];
  }

  sdata[threadIdx.y][k] = mySum;

  __syncthreads();

  if (k < 4) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata[threadIdx.y] + k;
    *smem = mySum = mySum + smem[4];
    *smem = mySum = mySum + smem[2];
    *smem = mySum = mySum + smem[1];
  }

  if (k == 0 && i < col) {
    p[i2] = v[i2] + mySum;
  }
}

template <typename real>
__global__ void kernel1(const real* sy, const int col, const real* v,
                        const int iPitch, real* p) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  const int k = threadIdx.x;

  volatile __shared__ real sdata[4][9];

  real mySum = 0;
  real pre = 0;

  if (i < col) {
    real syii = 1.0 / sy[i * iPitch + i];
    pre = -v[i] * syii;

    if (k > i && k < col) {
      mySum = sy[k * iPitch + i] * p[col + k] * syii;
    }
  }

  sdata[threadIdx.y][k] = mySum;

  __syncthreads();

  if (k < 4) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata[threadIdx.y] + k;
    *smem = mySum = mySum + smem[4];
    *smem = mySum = mySum + smem[2];
    *smem = mySum = mySum + smem[1];
  }

  if (k == 0 && i < col) {
    p[i] = pre + mySum;
  }
}

template <typename real>
void prog0(const real* sy, const int& col, const int& iPitch, const real* v,
           real* p, const cudaStream_t& st) {
  int nblocks = iDivUp(col, 4);

  if (col <= 1) {
    if (!st) {
      cudaMemcpy(p + col, v + col, sizeof(real), cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(p + col, v + col, sizeof(real), cudaMemcpyDeviceToDevice,
                      st);
    }
    return;
  }

  if (!st) {
    kernel0<<<nblocks, dim3(8, 4)>>>(sy, col, v, iPitch, p);
  } else {
    kernel0<<<nblocks, dim3(8, 4), 0, st>>>(sy, col, v, iPitch, p);
  }
}

template <typename real>
void prog1(const real* wt, const int& col, const int& iPitch, const real* v,
           real* p, cublasContext* cublas_handle, const cudaStream_t& st) {
  if (st) cublasSetStream(cublas_handle, st);

  cublasRtrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
              CUBLAS_DIAG_NON_UNIT, col, wt, iPitch, p + col, 1);
  cublasRtrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
              CUBLAS_DIAG_NON_UNIT, col, wt, iPitch, p + col, 1);
}

template <typename real>
void prog2(const real* sy, real* wt, const int& col, const int& iPitch,
           const real* v, real* p, const cudaStream_t& st) {
  int nblocks = iDivUp(col, 4);

  if (!st) {
    kernel1<<<nblocks, dim3(8, 4)>>>(sy, col, v, iPitch, p);
  } else {
    kernel1<<<nblocks, dim3(8, 4), 0, st>>>(sy, col, v, iPitch, p);
  }
}

#define INST_HELPER(real)                                                     \
  template void prog0<real>(const real*, const int&, const int&, const real*, \
                            real*, const cudaStream_t&);                      \
  template void prog1<real>(const real*, const int&, const int&, const real*, \
                            real*, cublasContext*, const cudaStream_t&);      \
  template void prog2<real>(const real*, real*, const int&, const int&,       \
                            const real*, real*, const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace bmv
};  // namespace cuda
};  // namespace lbfgsbcuda