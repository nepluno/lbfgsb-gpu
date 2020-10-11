/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

#include "culbfgsb/culbfgsb.h"
#include "examples/dsscfg/dsscfg_cpu.h"
#include "examples/dsscfg/dsscfg_cuda.h"

int g_nx = 256;
int g_ny = 256;

#define g_lambda 1.0

// test CPU mode
template <typename real>
real test_dsscfg_cpu() {
  // initialize LBFGSB option
  LBFGSB_CUDA_OPTION<real> lbfgsb_options;

  lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
  lbfgsb_options.mode = LCM_NO_ACCELERATION;
  lbfgsb_options.eps_f = static_cast<real>(1e-8);
  lbfgsb_options.eps_g = static_cast<real>(1e-8);
  lbfgsb_options.eps_x = static_cast<real>(1e-8);
  lbfgsb_options.max_iteration = 1000;

  // initialize LBFGSB state
  LBFGSB_CUDA_STATE<real> state;
  memset(&state, 0, sizeof(state));
  real* assist_buffer_cpu = nullptr;

  real minimal_f = std::numeric_limits<real>::max();
  // setup callback function that evaluate function value and its gradient
  state.m_funcgrad_callback = [&assist_buffer_cpu, &minimal_f](
                                  real* x, real& f, real* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<real>& summary) {
    dsscfg_cpu<real>(g_nx, g_ny, x, f, g, &assist_buffer_cpu, 'FG', g_lambda);
    if (summary.num_iteration % 100 == 0) {
      std::cout << "CPU iteration " << summary.num_iteration << " F: " << f
                << std::endl;    
    }

    minimal_f = fmin(minimal_f, f);
    return 0;
  };

  // initialize CPU buffers
  int N_elements = g_nx * g_ny;

  real* x = new real[N_elements];
  real* g = new real[N_elements];

  real* xl = new real[N_elements];
  real* xu = new real[N_elements];

  // in this example, we don't have boundaries
  memset(xl, 0, N_elements * sizeof(xl[0]));
  memset(xu, 0, N_elements * sizeof(xu[0]));

  // initialize starting point
  real f_init = std::numeric_limits<real>::max();
  dsscfg_cpu<real>(g_nx, g_ny, x, f_init, nullptr, &assist_buffer_cpu, 'XS',
             g_lambda);

  // initialize number of bounds (0 for this example)
  int* nbd = new int[N_elements];
  memset(nbd, 0, N_elements * sizeof(nbd[0]));

  LBFGSB_CUDA_SUMMARY<real> summary;
  memset(&summary, 0, sizeof(summary));

  // call optimization
  auto start_time = std::chrono::steady_clock::now();
  lbfgsbcuda::lbfgsbminimize<real>(N_elements, state, lbfgsb_options, x, nbd,
                                   xl, xu, summary);
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Timing: "
            << (std::chrono::duration<real, std::milli>(end_time - start_time)
                    .count() /
                static_cast<real>(summary.num_iteration))
            << " ms / iteration" << std::endl;

  // release allocated memory
  delete[] x;
  delete[] g;
  delete[] xl;
  delete[] xu;
  delete[] nbd;
  delete[] assist_buffer_cpu;

  return minimal_f;
}

// test CUDA mode
template <typename real>
real test_dsscfg_cuda() {
  // initialize LBFGSB option
  LBFGSB_CUDA_OPTION<real> lbfgsb_options;

  lbfgsbcuda::lbfgsbdefaultoption<real>(lbfgsb_options);
  lbfgsb_options.mode = LCM_CUDA;
  lbfgsb_options.eps_f = static_cast<real>(1e-8);
  lbfgsb_options.eps_g = static_cast<real>(1e-8);
  lbfgsb_options.eps_x = static_cast<real>(1e-8);
  lbfgsb_options.max_iteration = 1000;

  // initialize LBFGSB state
  LBFGSB_CUDA_STATE<real> state;
  memset(&state, 0, sizeof(state));
  real* assist_buffer_cuda = nullptr;
  cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
  if (CUBLAS_STATUS_SUCCESS != stat) {
    std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
    exit(0);
  }

  real minimal_f = std::numeric_limits<real>::max();
  // setup callback function that evaluate function value and its gradient
  state.m_funcgrad_callback = [&assist_buffer_cuda, &minimal_f](
                                  real* x, real& f, real* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<real>& summary) {
    dsscfg_cuda<real>(g_nx, g_ny, x, f, g, &assist_buffer_cuda, 'FG', g_lambda);
    if (summary.num_iteration % 100 == 0) {
      std::cout << "CUDA iteration " << summary.num_iteration << " F: " << f
                << std::endl;
    }
    minimal_f = fmin(minimal_f, f);
    return 0;
  };

  // initialize CUDA buffers
  int N_elements = g_nx * g_ny;

  real* x = nullptr;
  real* g = nullptr;
  real* xl = nullptr;
  real* xu = nullptr;
  int* nbd = nullptr;

  cudaMalloc(&x, N_elements * sizeof(x[0]));
  cudaMalloc(&g, N_elements * sizeof(g[0]));

  cudaMalloc(&xl, N_elements * sizeof(xl[0]));
  cudaMalloc(&xu, N_elements * sizeof(xu[0]));

  cudaMemset(xl, 0, N_elements * sizeof(xl[0]));
  cudaMemset(xu, 0, N_elements * sizeof(xu[0]));

  // initialize starting point
  real f_init = std::numeric_limits<real>::max();
  dsscfg_cuda<real>(g_nx, g_ny, x, f_init, g, &assist_buffer_cuda, 'XS',
                    g_lambda);

  // initialize number of bounds
  cudaMalloc(&nbd, N_elements * sizeof(nbd[0]));
  cudaMemset(nbd, 0, N_elements * sizeof(nbd[0]));

  LBFGSB_CUDA_SUMMARY<real> summary;
  memset(&summary, 0, sizeof(summary));

  // call optimization
  auto start_time = std::chrono::steady_clock::now();
  lbfgsbcuda::lbfgsbminimize<real>(N_elements, state, lbfgsb_options, x, nbd,
                                   xl, xu, summary);
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Timing: "
            << (std::chrono::duration<real, std::milli>(end_time - start_time)
                    .count() /
                static_cast<real>(summary.num_iteration))
            << " ms / iteration" << std::endl;

  // release allocated memory
  cudaFree(x);
  cudaFree(g);
  cudaFree(xl);
  cudaFree(xu);
  cudaFree(nbd);
  cudaFree(assist_buffer_cuda);

  // release cublas
  cublasDestroy(state.m_cublas_handle);
  return minimal_f;
}

int main(int argc, char* argv[]) {
  std::cout << "Begin testing DSSCFG on the CPU (double precision)"
            << std::endl;
  double min_f_cpu_dbl = test_dsscfg_cpu<double>();
    auto cpu_dbl_start = std::chrono::steady_clock::now();

  std::cout << "Begin testing DSSCFG with CUDA (double precision)" << std::endl;
  double min_f_gpu_dbl = test_dsscfg_cuda<double>();

  if (fabs(min_f_cpu_dbl - min_f_gpu_dbl) < 1e-4) {
    std::cout << "Passed!" << std::endl;
  } else {
    std::cout << "Failed: CPU result " << min_f_cpu_dbl << ", CUDA result "
              << min_f_gpu_dbl
              << ", error: " << fabs(min_f_cpu_dbl - min_f_gpu_dbl)
              << std::endl;
  }

  std::cout << "Begin testing DSSCFG on the CPU (single precision)"
            << std::endl;
  float min_f_cpu_sgl = test_dsscfg_cpu<float>();

  std::cout << "Begin testing DSSCFG with CUDA (single precision)" << std::endl;
  float min_f_gpu_sgl = test_dsscfg_cuda<float>();

  if (fabsf(min_f_cpu_sgl - min_f_gpu_sgl) < 1e-3f) {
    std::cout << "Passed!" << std::endl;
  } else {
    std::cout << "Failed: CPU result " << min_f_cpu_sgl << ", CUDA result "
              << min_f_gpu_sgl
              << ", error: " << fabs(min_f_cpu_sgl - min_f_gpu_sgl)
              << std::endl;
  }

  return 0;
}
