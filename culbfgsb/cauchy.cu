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
namespace cauchy {

template <int bx, typename real>
__global__ void kernel0(int n, const real* g, const int* nbd, real* t,
                        const real* x, const real* u, const real* l,
                        int* iwhere, const real machinemaximum) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int tid = threadIdx.x;
  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n) {
    int iwi = iwhere[i];
    if (iwi != 3 && iwi != -1) {
      real neggi = -g[i];
      int nbdi = nbd[i];

      real tl = 0;
      real tu = 0;

      if (nbdi <= 2) {
        tl = x[i] - l[i];
      }
      if (nbdi >= 2) {
        tu = u[i] - x[i];
      }

      if (nbdi <= 2 && tl <= 0 && neggi <= 0) {
        iwi = 1;
      } else if (nbdi >= 2 && tu <= 0 && neggi >= 0) {
        iwi = 2;
      } else if (neggi == 0) {
        iwi = -3;
      } else {
        iwi = 0;
      }

      iwhere[i] = iwi;

      if ((iwi != 0 && iwi != -1) || neggi == 0) {
        mySum = machinemaximum;
      } else {
        if (nbdi <= 2 && nbdi != 0 && neggi < 0) {
          mySum = tl / (-neggi);
        } else if (nbdi >= 2 && neggi > 0) {
          mySum = tu / neggi;
        } else {
          mySum = machinemaximum;
        }
      }
    } else {
      mySum = machinemaximum;
    }
  } else {
    mySum = machinemaximum;
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

  if (tid == 0) t[blockIdx.x] = mySum;
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
    mySum = machinemaximum;

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

template <int bx, typename real>
__global__ void kernel1(const int n, const real* g, real* buf_s_r,
                        const int* iwhere) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int tid = threadIdx.x;
  volatile __shared__ real sdata[bx];

  real mySum;

  if (i >= n) {
    mySum = 0;
  } else {
    int iwi = iwhere[i];
    if (iwi != 0 && iwi != -1) {
      mySum = 0;
    } else {
      real neggi = g[i];
      mySum = -neggi * neggi;
    }
  }

  sdata[tid] = mySum;
  __syncthreads();

  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) buf_s_r[blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel20(const int n, const int head, const int m,
                         const int col, const int iPitch, const int oPitch,
                         const real* g, real* buf_array_p, const real* wy,
                         const real* ws, const int* iwhere) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n) {
    int iwi = iwhere[i];
    if (iwi != 0 && iwi != -1) {
      mySum = 0;
    } else {
      real neggi = -g[i];

      real p0;
      if (j < col) {
        int pointr = Modular((head + j), m);
        p0 = wy[i * iPitch + pointr];
      } else {
        int pointr = Modular((head + j - col), m);
        p0 = ws[i * iPitch + pointr];
      }

      mySum = p0 * neggi;
    }
  } else {
    mySum = 0;
  }

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) buf_array_p[j * oPitch + blockIdx.x] = mySum;
}

template <int bx, typename real>
__global__ void kernel21(const int n, const int iPitch, const int oPitch,
                         const real* buf_in, real* buf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int tid = threadIdx.x;

  volatile __shared__ real sdata[bx];

  real mySum;

  if (i < n)
    mySum = buf_in[j * iPitch + i];
  else
    mySum = 0;

  sdata[tid] = mySum;
  __syncthreads();
  if (bx > 512) {
    if (tid < 512) {
      sdata[tid] = mySum = (mySum + sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (bx > 256) {
    if (tid < 256) {
      sdata[tid] = mySum = (mySum + sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (bx > 128) {
    if (tid < 128) {
      sdata[tid] = mySum = (mySum + sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (bx > 64) {
    if (tid < 64) {
      sdata[tid] = mySum = (mySum + sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < min(bx / 2, 32)) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile real* smem = sdata + tid;
    if (bx > 32) {
      *smem = mySum = mySum + smem[32];
    }
    if (bx > 16) {
      *smem = mySum = mySum + smem[16];
    }
    if (bx > 8) {
      *smem = mySum = mySum + smem[8];
    }
    if (bx > 4) {
      *smem = mySum = mySum + smem[4];
    }
    if (bx > 2) {
      *smem = mySum = mySum + smem[2];
    }
    if (bx > 1) {
      *smem = mySum = mySum + smem[1];
    }
  }

  if (tid == 0) {
    buf_out[j * oPitch + blockIdx.x] = mySum;
  }
}

template <typename real>
__global__ void kernel22(const int n, real* p, const real theta) {
  const int i = threadIdx.x;

  if (i >= n) return;

  p[i] *= theta;
}

template <typename real>
__global__ void kernel4(const int col2, const real* p, real* c,
                        const real dtm) {
  const int i = threadIdx.x;

  if (i >= col2) return;

  c[i] = p[i] * dtm;
}

template <typename real>
__global__ void kernel3(const int n, const real* x, const real* g, real* xcp,
                        real* xcpb, const real dtm, const int* iwhere) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  real inc;
  int iwi = iwhere[i];
  if (iwi != 0 && iwi != -1) {
    inc = 0;
  } else {
    real neggi = -g[i];
    inc = neggi * dtm;
  }
  real res = x[i] + inc;
  xcp[i] = res;
  xcpb[i] = res;
}

template <typename real>
void prog0(const int& n, const real* x, const real* l, const real* u,
           const int* nbd, const real* g, real* t, real* xcp, real* xcpb,
           const int& m, const real* wy, const real* ws, const real* sy,
           const int iPitch, real* wt, const real& theta, const int& col,
           const int& head, real* p, real* c, real* v, int& nint,
           const real& sbgnrm, real* buf_s_r, real* buf_array_p, int* iwhere,
           const int& iPitch_normal, const real& machinemaximum,
           cublasHandle_t cublas_handle, const cudaStream_t* streamPool) {
  debugSync();

  if (sbgnrm <= 0) {
    cudaMemcpyAsync(xcp, x, n * sizeof(real), cudaMemcpyDeviceToDevice);
    return;
  }

  if (col > 0) cudaMemsetAsync(p, 0, col * 2 * sizeof(real));

  real* vec_h;
  real* vec_d;

  cutilSafeCall(cudaHostAlloc(&vec_h, 3 * sizeof(real), cudaHostAllocMapped));
  cutilSafeCall(cudaHostGetDevicePointer(&vec_d, vec_h, 0));

  real* bkmin_d = vec_d;
  real* f1_d = vec_d + 1;

  real* bkmin_h = vec_h;
  real* f1_h = vec_h + 1;
  real* fd_h = vec_h + 2;

  int nblock0 = n;
  int mi = log2Up(nblock0);
  int nblock1 = iDivUp2(nblock0, mi);

  real* output0 = (nblock1 == 1) ? bkmin_d : t;
  real* output1 = (nblock1 == 1) ? f1_d : buf_s_r;
  real* output2 = (nblock1 == 1) ? p : buf_array_p;

  dynamicCall(kernel0, mi, real, nblock1, 1, streamPool[0],
              (nblock0, g, nbd, output0, x, u, l, iwhere, machinemaximum));

  dynamicCall(kernel1, mi, real, nblock1, 1, streamPool[0],
              (nblock0, g, output1, iwhere));

  int op20 = (nblock1 == 1) ? 1 : iPitch_normal;

  if (col > 0) {
    dynamicCall(
        kernel20, mi, real, nblock1, col * 2, streamPool[0],
        (nblock0, head, m, col, iPitch, op20, g, output2, wy, ws, iwhere));
  }
  nblock0 = nblock1;

  while (nblock0 > 1) {
    nblock1 = iDivUp2(nblock0, mi);

    real* input0 = output0;
    real* input1 = output1;
    real* input2 = output2;

    output0 = (nblock1 == 1) ? bkmin_d : (output0 + nblock0);
    output1 = (nblock1 == 1) ? f1_d : (output1 + nblock0);
    output2 = (nblock1 == 1) ? p : (output2 + nblock0);

    dynamicCall(kernel01, mi, real, nblock1, 1, streamPool[0],
                (nblock0, input0, output0, machinemaximum));

    dynamicCall(kernel21, mi, real, nblock1, 1, streamPool[1],
                (nblock0, 1, 1, input1, output1));

    int op20 = (nblock1 == 1) ? 1 : iPitch_normal;
    if (col > 0) {
      dynamicCall(kernel21, mi, real, nblock1, col * 2, streamPool[2],
                  (nblock0, iPitch_normal, op20, input2, output2));
    }

    nblock0 = nblock1;
  }

  if (col > 0 && theta != 1) {
    debugSync();
    kernel22<real><<<dim3(1), dim3(col), 0, streamPool[2]>>>(col, p + col, theta);

    debugSync();
  }

  *fd_h = 0;

  if (col > 0) {
    bmv::prog0<real>(sy, col, iPitch, p, v, streamPool[2]);

    debugSync();
    bmv::prog1<real>(wt, col, iPitch, p, v, cublas_handle, streamPool[2]);

    debugSync();
    bmv::prog2<real>(sy, wt, col, iPitch, p, v, streamPool[2]);

    debugSync();
    cublasSetStream(cublas_handle, streamPool[2]);

    cublasRdot<real>(cublas_handle, col * 2, v, 1, p, 1, fd_h);

    cublasSetStream(cublas_handle, NULL);
  }

  cutilSafeCall(cudaDeviceSynchronize());

  real f2 = -theta * *f1_h - *fd_h;
  real dt = -*f1_h / f2;

  real dtm = std::min(*bkmin_h, dt);
  dtm = std::max(static_cast<real>(0.0), dtm);

  kernel3<real><<<dim3(iDivUp(n, 512)), dim3(512), 0, streamPool[0]>>>(
      n, x, g, xcp, xcpb, dtm, iwhere);

  if (col > 0) {
    kernel4<real>
        <<<dim3(1), dim3(col * 2), 0, streamPool[1]>>>(col * 2, p, c, dtm);
  }
}

#define INST_HELPER(real)                                                     \
  template void prog0<real>(                                                  \
      const int&, const real*, const real*, const real*, const int*,          \
      const real*, real*, real*, real*, const int&, const real*, const real*, \
      const real*, const int, real*, const real&, const int&, const int&,     \
      real*, real*, real*, int&, const real&, real*, real*, int*, const int&, \
      const real&, cublasHandle_t, const cudaStream_t*);

INST_HELPER(double);
INST_HELPER(float);
};  // namespace cauchy
};  // namespace cuda
};  // namespace lbfgsbcuda