/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "culbfgsb/lbfgsbcuda.h"

#ifdef __INTELLISENSE__
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
#endif

namespace lbfgsbcuda {
namespace cuda {
namespace active {
template <typename real>
__global__ void kernel0(const int n, const real* l, const real* u,
                        const int* nbd, real* x, int* iwhere) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int nbdi = nbd[i];
  real xi = x[i];
  real li = l[i];
  real ui = u[i];
  int iwi = -1;

  if (nbdi > 0) {
    if (nbdi <= 2) {
      xi = maxr(xi, li);
    } else {
      xi = minr(xi, ui);
    }
  }

  if (nbdi == 2 && ui - li <= 0) {
    iwi = 3;
  } else if (nbdi != 0) {
    iwi = 0;
  }

  x[i] = xi;
  iwhere[i] = iwi;
}

template <typename real>
void prog0(const int& n, const real* l, const real* u, const int* nbd, real* x,
           int* iwhere) {
  kernel0<real><<<dim3(iDivUp(n, 512)), dim3(512)>>>(n, l, u, nbd, x, iwhere);

  debugSync();
}

template void prog0<float>(const int&, const float*, const float*,
                           const int*, float*, int*);
template void prog0<double>(const int&, const double*, const double*,
                            const int*, double*, int*);

};  // namespace active
};  // namespace cuda
};  // namespace lbfgsbcuda