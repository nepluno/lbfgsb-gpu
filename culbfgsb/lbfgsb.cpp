/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LBFGSB_CUDA_EXPORTS
#include <fstream>
#include <iostream>

#include "culbfgsb/cutil_inline.h"
#include "culbfgsb/lbfgsbcpu.h"
#include "culbfgsb/lbfgsbcuda.h"

namespace lbfgsbcuda {
template <typename real>
LBFGSB_CUDA_FUNCTION void lbfgsbminimize(const int& n,
                                         const LBFGSB_CUDA_STATE<real>& state,
                                         const LBFGSB_CUDA_OPTION<real>& option,
                                         real* x, const int* nbd, const real* l,
                                         const real* u,
                                         LBFGSB_CUDA_SUMMARY<real>& summary) {
  switch (option.mode) {
    case LCM_NO_ACCELERATION:
      cpu::lbfgsbminimize<real>(n, state, option, x, nbd, l, u, summary);
      break;
    case LCM_CUDA:
      cuda::lbfgsbminimize<real>(n, state, option, x, nbd, l, u, summary);
      break;
    default:
      std::cerr << "ERROR: INVALID MODE" << std::endl;
      break;
  }
}

template <typename real>
LBFGSB_CUDA_FUNCTION void lbfgsbdefaultoption(
    LBFGSB_CUDA_OPTION<real>& option) {
  option.eps_f = option.eps_g = option.eps_x = 1e-15;
  option.hessian_approximate_dimension = 8;
  option.machine_epsilon = 1e-15;
  option.machine_maximum = std::numeric_limits<real>::max();
  option.max_iteration = 1000;
  option.mode = LCM_CUDA;
  option.step_scaling = 1.0;
}

template <>
LBFGSB_CUDA_FUNCTION void lbfgsbdefaultoption<float>(
    LBFGSB_CUDA_OPTION<float>& option) {
  option.eps_f = option.eps_g = option.eps_x = 1e-15f;
  option.hessian_approximate_dimension = 8;
  option.machine_epsilon = 1e-8f;
  option.machine_maximum = std::numeric_limits<float>::max();
  option.max_iteration = 1000;
  option.mode = LCM_CUDA;
  option.step_scaling = 1.0f;
}

#define INST_HELPER(real)                                              \
  template LBFGSB_CUDA_FUNCTION void lbfgsbminimize<real>(             \
      const int&, const LBFGSB_CUDA_STATE<real>&,                      \
      const LBFGSB_CUDA_OPTION<real>&, real*, const int*, const real*, \
      const real*, LBFGSB_CUDA_SUMMARY<real>&);                        \
  template LBFGSB_CUDA_FUNCTION void lbfgsbdefaultoption<real>(        \
      LBFGSB_CUDA_OPTION<real> & option);

INST_HELPER(double);
INST_HELPER(float);

}  // namespace lbfgsbcuda