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
namespace lnsrlb {

template <int bx, typename real>
__global__ void kernel00(int n, const real* d, const int* nbd, const real* u,
                         const real* x, const real* l, real* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int tid = threadIdx.x;
  volatile __shared__ real sdata[bx];

  real mySum = 1e10;
  if (i < n) {
    real a1 = d[i];
    int nbdi = nbd[i];
    if (nbdi != 0) {
      real xi = x[i];
      real a2;
      if (a1 > 0) {
        a2 = u[i] - xi;
      } else {
        a2 = xi - l[i];
      }
      a2 = maxr(static_cast<real>(0.0), a2);

      mySum = absr(a2 / a1);
    }
  }

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = minr(mySum, smem[32]);
    }
    if (bx > 16) {
      *smem = mySum = minr(mySum, smem[16]);
    }
    if (bx > 8) {
      *smem = mySum = minr(mySum, smem[8]);
    }
    if (bx > 4) {
      *smem = mySum = minr(mySum, smem[4]);
    }
    if (bx > 2) {
      *smem = mySum = minr(mySum, smem[2]);
    }
    if (bx > 1) {
      *smem = mySum = minr(mySum, smem[1]);
    }
  }

  if (tid == 0) output[blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel01(int n, const real* buf_in, real* buf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n)
    mySum = buf_in[i];
  else
    mySum = 1e10;

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = minr(mySum, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = minr(mySum, smem[32]);
    }
    if (bx > 16) {
      *smem = mySum = minr(mySum, smem[16]);
    }
    if (bx > 8) {
      *smem = mySum = minr(mySum, smem[8]);
    }
    if (bx > 4) {
      *smem = mySum = minr(mySum, smem[4]);
    }
    if (bx > 2) {
      *smem = mySum = minr(mySum, smem[2]);
    }
    if (bx > 1) {
      *smem = mySum = minr(mySum, smem[1]);
    }
  }

  if (tid == 0) {
    buf_out[blockIdx.x] = mySum;
  }
}

template <typename real>
__global__ void kernel2(int n, real* x, real* d, const real* t,
                        const real stp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  real u = stp * d[i];
  x[i] = t[i] + u;
}

template <typename real>
void prog0(int n, const real* d, const int* nbd, const real* u, const real* x,
           const real* l, real* buf_s_r, real* stpmx_host, real* stpmx_dev,
           const cudaStream_t& stream) {
  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? stpmx_dev : buf_s_r;
  dynamicCall(kernel00, mi, real, nblock1, 1, stream, (n, d, nbd, u, x, l, output));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? stpmx_dev : (output + nblock0);
    dynamicCall(kernel01, mi, real, nblock1, 1, stream, (nblock0, input, output));

    nblock0 = nblock1;
  }
}

template <typename real>
void prog2(int n, real* x, real* d, const real* t, const real stp,
           const cudaStream_t& stream) {
  kernel2<real><<<dim3(iDivUp(n, 512)), dim3(512), 0, stream>>>(n, x, d, t, stp);
}

#define INST_HELPER(real)                                                  \
  template void prog0<real>(int, const real*, const int*, const real*,     \
                            const real*, const real*, real*, real*, real*, \
                            const cudaStream_t&);                          \
  template void prog2<real>(int, real*, real*, const real*, const real,    \
                            const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);

};  // namespace lnsrlb
};  // namespace cuda
};  // namespace lbfgsbcuda