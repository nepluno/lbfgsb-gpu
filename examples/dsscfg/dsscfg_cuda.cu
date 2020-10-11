/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "examples/dsscfg/dsscfg_cuda.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#endif

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

template <typename real>
__global__ void dsscfg_kernel_init(int nx, int ny, real* x, real lambda) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nx || j >= ny) {
    return;
  }
  real hx = 1.0 / static_cast<real>(nx + 1);
  real hy = 1.0 / static_cast<real>(ny + 1);
  real temp1 = lambda / (lambda + 1.0);

  real temp = static_cast<real>(min(j, ny - j)) * hy;
  int k = nx * j + i;
  x[k] = temp1 * sqrt(fmin(static_cast<real>(min(i, nx - i)) * hx, temp));
}

template <typename real>
__global__ void dsscfg_kernel_func_grad(int nx, int ny, const real* x,
                                        real* fquad, real* fexp,
                                        real* fgrad, real* fgrad_r,
                                        real* fgrad_t, real* fgrad_l,
                                        real* fgrad_b, real lambda,
                                        bool feval, bool geval) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nx || j >= ny) {
    return;
  }

  int k = nx * j + i;
  real hx = 1.0 / static_cast<real>(nx + 1);
  real hy = 1.0 / static_cast<real>(ny + 1);
  real v = 0.0;
  real vr = 0.0;
  real vt = 0.0;
  v = x[k];
  if (i != nx - 1) {
    vr = x[k + 1];
  }
  if (j != ny - 1) {
    vt = x[k + nx];
  }
  real dvdx = (vr - v) / hx;
  real dvdy = (vt - v) / hy;
  real expv = exp(v);
  real expvr = exp(vr);
  real expvt = exp(vt);

  real local_fgrad = 0.0;
  real local_fquad = 0.0;
  real local_fexp = 0.0;

  if (feval) {
    local_fquad = fquad[k] + dvdx * dvdx + dvdy * dvdy;
    local_fexp = fexp[k] - lambda * (expv + expvr + expvt) / 3.0;
  }
  if (geval) {
    local_fgrad = fgrad[k] - dvdx / hx - dvdy / hy - lambda * expv / 3.0;
    if (i != nx - 1) {
      fgrad_r[k + 1] = dvdx / hx - lambda * expvr / 3.0;
    }
    if (j != ny - 1) {
      fgrad_t[k + nx] = dvdy / hy - lambda * expvt / 3.0;
    }
  }

  //
  //     Computation of the function and the gradient over the upper
  //     triangular elements.  The trapezoidal rule is used to estimate
  //     the integral of the exponential term.
  //

  real vb = 0.0;
  real vl = 0.0;

  if (j != 0) {
    vb = x[k - nx];
  }
  if (i != 0) {
    vl = x[k - 1];
  }

  dvdx = (v - vl) / hx;
  dvdy = (v - vb) / hy;
  real expvb = exp(vb);
  real expvl = exp(vl);
  expv = exp(v);
  if (feval) {
    local_fquad += dvdx * dvdx + dvdy * dvdy;
    local_fexp -= lambda * (expvb + expvl + expv) / 3.0;

    fquad[k] = local_fquad;
    fexp[k] = local_fexp;
  }
  if (geval) {
    if (j != 0) {
      fgrad_b[k - nx] = -dvdy / hy - lambda * expvb / 3.0;
    }
    if (i != 0) {
      fgrad_l[k - 1] = -dvdx / hx - lambda * expvl / 3.0;
    }
    local_fgrad += dvdx / hx + dvdy / hy - lambda * expv / 3.0;

    fgrad[k] = local_fgrad;
  }
}

template <typename real>
__global__ void dsscfg_kernel_scale_grad(int nx, int ny, const real* fgrad_r,
                                         const real* fgrad_t,
                                         const real* fgrad_l,
                                         const real* fgrad_b, real* fgrad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= nx || j >= ny) {
    return;
  }

  int k = nx * j + i;
  real hx = 1.0 / (real)(nx + 1);
  real hy = 1.0 / (real)(ny + 1);
  real area = 0.5 * hx * hy;

  fgrad[k] =
      (fgrad[k] + fgrad_b[k] + fgrad_l[k] + fgrad_r[k] + fgrad_t[k]) * area;
}

template <typename real>
void dsscfg_cuda(int const& nx, int const& ny, real* x, real& f,
                 real* fgrad, real** assist_buffer, int task,
                 real const& lambda) {
  dim3 blocksize = {16U, 16U, 1U};
  dim3 gridsize = {(nx + 15U) / 16U, (ny + 15U) / 16U, 1U};

  if (task == 'XS') {
    dsscfg_kernel_init<real><<<gridsize, blocksize>>>(nx, ny, x, lambda);
    cudaMalloc(assist_buffer, nx * ny * 6 * sizeof(real));
    return;
  }

  bool feval = task == 'F' || task == 'FG';
  bool geval = task == 'G' || task == 'FG';

  real* fquad = nullptr;
  real* fexp = nullptr;
  real* fgrad_r = nullptr;
  real* fgrad_t = nullptr;
  real* fgrad_l = nullptr;
  real* fgrad_b = nullptr;

  if (feval) {
    fquad = (*assist_buffer) + nx * ny * 4;
    fexp = (*assist_buffer) + nx * ny * 5;
  }

  if (geval) {
    fgrad_r = (*assist_buffer);
    fgrad_t = (*assist_buffer) + nx * ny;
    fgrad_l = (*assist_buffer) + nx * ny * 2;
    fgrad_b = (*assist_buffer) + nx * ny * 3;

    cudaMemsetAsync(fgrad, 0, nx * ny * sizeof(real));
  }

  if (feval || geval) {
    cudaMemsetAsync(*assist_buffer, 0, nx * ny * 6 * sizeof(real));
    dsscfg_kernel_func_grad<real><<<gridsize, blocksize>>>(
        nx, ny, x, fquad, fexp, fgrad, fgrad_r, fgrad_t, fgrad_l, fgrad_b,
        lambda, feval, geval);
  }

  if (feval) {
    thrust::device_ptr<real> dev_fquad = thrust::device_pointer_cast(fquad);
    thrust::device_ptr<real> dev_fexp = thrust::device_pointer_cast(fexp);

    real sum_fquad = thrust::reduce(dev_fquad, dev_fquad + nx * ny);
    real sum_fexp = thrust::reduce(dev_fexp, dev_fexp + nx * ny);

    real hx = 1.0 / static_cast<real>(nx + 1);
    real hy = 1.0 / static_cast<real>(ny + 1);
    real area = 0.5 * hx * hy;

    f = area * (.5 * sum_fquad + sum_fexp);
  }

  if (geval) {
    dsscfg_kernel_scale_grad<real><<<gridsize, blocksize>>>(nx, ny, fgrad_r, fgrad_t,
                                                      fgrad_l, fgrad_b, fgrad);
  }
}

#define INST_HELPER(real)                                                     \
  template void dsscfg_cuda<real>(int const& nx, int const& ny, real* x,      \
                                  real& f, real* fgrad, real** assist_buffer, \
                                  int task, real const& lambda);

INST_HELPER(float);
INST_HELPER(double);
