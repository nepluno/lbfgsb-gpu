/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "lbfgsbcuda.h"

namespace lbfgsbcuda {
namespace cuda {
namespace minimize {

template <typename real>
__global__ void kernel0(int n, real* d, const real stp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  d[i] *= stp;
}

template <typename real>
__global__ void kernel1(int n, const real* a, const real* b, real* c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  c[i] = a[i] - b[i];
}

template <typename real>
__global__ void kernel2(int n, real* xdiff, real* xold, const real* x) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  real xi = x[i];
  xdiff[i] = xold[i] - xi;
  xold[i] = xi;
}

template <typename real>
void vsub_v(const int n, const real* a, const real* b, real* c,
            const cudaStream_t& stream) {
  kernel1<real><<<iDivUp(n, 512), 512, 0, stream>>>(n, a, b, c);
}

template <typename real>
void vdiffxchg_v(const int n, real* xdiff, real* xold, const real* x,
                 const cudaStream_t& stream) {
  kernel2<real><<<iDivUp(n, 512), 512, 0, stream>>>(n, xdiff, xold, x);
}

template <typename real>
void vmul_v(const int n, real* d, const real stp, const cudaStream_t& stream) {
  kernel0<real><<<iDivUp(n, 512), 512, 0, stream>>>(n, d, stp);
}

template <typename real>
void vdot_vv(const int n, const real* g, const real* d, real& gd,
             cublasHandle_t cublas_handle, const cudaStream_t& stream) {
  cublasSetStream(cublas_handle, stream);
  cublasRdot<real>(cublas_handle, n, g, 1, d, 1, &gd);
  cublasSetStream(cublas_handle, NULL);
}

#define INST_HELPER(real)                                                 \
  template void vsub_v<real>(const int, const real*, const real*, real*,  \
                             const cudaStream_t&);                        \
  template void vdiffxchg_v<real>(const int, real*, real*, const real*,   \
                                  const cudaStream_t&);                   \
  template void vmul_v<real>(const int, real*, const real,                \
                             const cudaStream_t&);                        \
  template void vdot_vv<real>(const int, const real*, const real*, real&, \
                              cublasHandle_t, const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace minimize
};  // namespace cuda
};  // namespace lbfgsbcuda