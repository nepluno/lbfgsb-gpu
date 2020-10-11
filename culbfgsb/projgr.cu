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
namespace projgr {
template <int bx, typename real>
__global__ void kernel0(const int n, const real* l, const real* u,
                        const int* nbd, const real* x, const real* g,
                        real* buf_n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  volatile __shared__ real sdata[bx];

  real mySum;

  if (i >= n) {
    mySum = sdata[tid] = 0;
  } else {
    real gi = g[i];
    int nbdi = nbd[i];
    if (nbdi != 0) {
      if (gi < 0) {
        if (nbdi >= 2) {
          gi = maxr(x[i] - u[i], gi);
        }
      } else {
        if (nbdi <= 2) {
          gi = minr(x[i] - l[i], gi);
        }
      }
    }
    mySum = sdata[tid] = absr(gi);
  }
  __syncthreads();

  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = maxr(mySum, smem[32]);
    }
    if (bx > 16) {
      *smem = mySum = maxr(mySum, smem[16]);
    }
    if (bx > 8) {
      *smem = mySum = maxr(mySum, smem[8]);
    }
    if (bx > 4) {
      *smem = mySum = maxr(mySum, smem[4]);
    }
    if (bx > 2) {
      *smem = mySum = maxr(mySum, smem[2]);
    }
    if (bx > 1) {
      *smem = mySum = maxr(mySum, smem[1]);
    }
  }

  if (tid == 0) buf_n[blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel01(const int n, const real* buf_in, real* buf_out,
                         const real machinemaximum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n)
    mySum = buf_in[i];
  else
    mySum = -machinemaximum;

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = maxr(mySum, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = maxr(mySum, smem[32]);
    }
    if (bx > 16) {
      *smem = mySum = maxr(mySum, smem[16]);
    }
    if (bx > 8) {
      *smem = mySum = maxr(mySum, smem[8]);
    }
    if (bx > 4) {
      *smem = mySum = maxr(mySum, smem[4]);
    }
    if (bx > 2) {
      *smem = mySum = maxr(mySum, smem[2]);
    }
    if (bx > 1) {
      *smem = mySum = maxr(mySum, smem[1]);
    }
  }

  if (tid == 0) {
    buf_out[blockIdx.x] = mySum;
  }
}

template <typename real>
void prog0(const int& n, const real* l, const real* u, const int* nbd,
           const real* x, const real* g, real* buf_n, real* sbgnrm,
           real* sbgnrm_dev, const real machinemaximum,
           const cudaStream_t& stream) {
  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output = (nblock1 == 1) ? sbgnrm_dev : buf_n;

  dynamicCall(kernel0, mi, real, nblock1, 1, stream,
              (n, l, u, nbd, x, g, output));

  nblock0 = nblock1;
  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input = output;

    output = (nblock1 == 1) ? sbgnrm_dev : (output + nblock0);

    dynamicCall(kernel01, mi, real, nblock1, 1, stream,
                (nblock0, input, output, machinemaximum));

    nblock0 = nblock1;
  }
}

#define INST_HELPER(real)                                                     \
  template void prog0<real>(const int&, const real*, const real*, const int*, \
                            const real*, const real*, real*, real*, real*,    \
                            const real, const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);
};  // namespace projgr
}  // namespace cuda
};  // namespace lbfgsbcuda