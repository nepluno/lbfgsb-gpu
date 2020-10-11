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

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "culbfgsb/lbfgsbcuda.h"

namespace lbfgsbcuda {
namespace cuda {
__global__ void kernel0(int* index, const int* iwhere, int* temp_ind1,
                        int* temp_ind2, int nfree, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  int k = index[i];
  int iwk = iwhere[k];
  int t1, t2;

  if (i < nfree && iwk > 0) {
    t1 = 1;
    t2 = 0;
  } else if (i >= nfree && iwk <= 0) {
    t1 = 0;
    t2 = 1;
  } else {
    t1 = t2 = 0;
  }

  temp_ind1[i] = t1;
  temp_ind2[i] = t2;
}

__global__ void kernel1(const int* index, const int* temp_ind1,
                        const int* temp_ind2, const int* temp_ind3,
                        const int* temp_ind4, int* indx2, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  int k = index[i];
  if (temp_ind1[i]) {
    indx2[n - temp_ind3[i]] = k;
  } else if (temp_ind2[i]) {
    indx2[temp_ind4[i] - 1] = k;
  }
}

__global__ void kernel2(const int* iwhere, int* temp_ind1, int* temp_ind2,
                        int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  int iwi = iwhere[i];
  if (iwi <= 0) {
    temp_ind1[i] = 1;
    temp_ind2[i] = 0;
  } else {
    temp_ind1[i] = 0;
    temp_ind2[i] = 1;
  }
}

__global__ void kernel3(int* index, const int* iwhere, const int* temp_ind1,
                        const int* temp_ind2, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n) return;

  int iwi = iwhere[i];
  if (iwi <= 0) {
    index[temp_ind1[i] - 1] = i;
  } else {
    index[n - temp_ind2[i]] = i;
  }
}

namespace freev {
void prog0(const int& n, int& nfree, int* index, int& nenter, int& ileave,
           int* indx2, const int* iwhere, bool& wrk, const bool& updatd,
           const bool& cnstnd, const int& iter, int* temp_ind1, int* temp_ind2,
           int* temp_ind3, int* temp_ind4) {
  nenter = -1;
  ileave = n;
  if (iter > 0 && cnstnd) {
    debugSync();
    kernel0<<<iDivUp(n, 512), 512>>>(index, iwhere, temp_ind1, temp_ind2, nfree,
                                     n);

    debugSync();

    thrust::device_ptr<int> dptr_ind1(temp_ind1);
    thrust::device_ptr<int> dptr_ind2(temp_ind2);
    thrust::device_ptr<int> dptr_ind3(temp_ind3);
    thrust::device_ptr<int> dptr_ind4(temp_ind4);

    thrust::inclusive_scan(dptr_ind1, dptr_ind1 + n, dptr_ind3);
    thrust::inclusive_scan(dptr_ind2, dptr_ind2 + n, dptr_ind4);

    debugSync();

    kernel1<<<iDivUp(n, 512), 512>>>(index, temp_ind1, temp_ind2, temp_ind3,
                                     temp_ind4, indx2, n);

    debugSync();

    cutilSafeCall(cudaMemcpy(&ileave, temp_ind3 + (n - 1), sizeof(int),
                             cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(&nenter, temp_ind4 + (n - 1), sizeof(int),
                             cudaMemcpyDeviceToHost));
    ileave = n - ileave;
    nenter = nenter - 1;
  }

  wrk = ileave < n || nenter >= 0 || updatd;

  debugSync();

  kernel2<<<iDivUp(n, 512), 512>>>(iwhere, temp_ind1, temp_ind2, n);

  debugSync();

  thrust::device_ptr<int> dptr_ind1(temp_ind1);
  thrust::device_ptr<int> dptr_ind2(temp_ind2);

  thrust::inclusive_scan(dptr_ind1, dptr_ind1 + n, dptr_ind1);
  thrust::inclusive_scan(dptr_ind2, dptr_ind2 + n, dptr_ind2);

  debugSync();

  kernel3<<<iDivUp(n, 512), 512>>>(index, iwhere, temp_ind1, temp_ind2, n);

  debugSync();

  cutilSafeCall(cudaMemcpy(&nfree, temp_ind1 + (n - 1), sizeof(int),
                           cudaMemcpyDeviceToHost));
}
};  // namespace freev
};  // namespace cuda
};  // namespace lbfgsbcuda