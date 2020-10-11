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
namespace dpofa {
#define CUDA_BLOCK_SIZE 16

template <typename real>
__global__ void cuda_chol_iter(real* m, int n, int boffset,
                               const real machineepsilon) {
  int k;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bsize = blockDim.x;
  __shared__ real b[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE + 1];
  b[x][y] = m[(x + boffset) * n + boffset + y];
  for (k = 0; k < bsize; k++) {
    __syncthreads();
    if (x == k) {
      if (b[x][x] < machineepsilon) b[x][x] = machineepsilon;
      real fac = sqrtr(b[x][x]);
      if (y >= x) {
        b[x][y] /= fac;
      }
    }
    __syncthreads();
    if (x > k && y >= x) {
      b[x][y] -= b[k][y] * b[k][x];
    }
  }
  __syncthreads();
  m[(boffset + x) * n + boffset + y] = b[x][y];
}

template <typename real>
void prog0(real* m, int n, int pitch, int boffset, const real machineepsilon,
           const cudaStream_t& st) {
  cuda_chol_iter<real><<<1, dim3(n, n), 0, st>>>(m, pitch, boffset, machineepsilon);
}

#define INST_HELPER(real)                                     \
  template void prog0<real>(real*, int, int, int, const real, \
                            const cudaStream_t&);

INST_HELPER(double);
INST_HELPER(float);

}  // namespace dpofa
}  // namespace cuda
};  // namespace lbfgsbcuda