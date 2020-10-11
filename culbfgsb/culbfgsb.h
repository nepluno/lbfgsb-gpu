/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CULBFGSB_CULBFGSB_H_
#define CULBFGSB_CULBFGSB_H_

#if defined (_WIN32) && defined (CULBFGSB_SHARED)
#define LBFGSB_CUDA_IMPORT __declspec(dllimport)
#define LBFGSB_CUDA_EXPORT __declspec(dllexport)
#else
#define LBFGSB_CUDA_IMPORT
#define LBFGSB_CUDA_EXPORT
#endif

#ifdef LBFGSB_CUDA_EXPORTS
#define LBFGSB_CUDA_FUNCTION LBFGSB_CUDA_EXPORT
#else
#define LBFGSB_CUDA_FUNCTION LBFGSB_CUDA_IMPORT
#endif

#include <cublas_v2.h>

#include <functional>

enum LBFGSB_CUDA_MODE { LCM_NO_ACCELERATION, LCM_CUDA };

template <typename real>
struct LBFGSB_CUDA_OPTION {
  real machine_epsilon;
  real machine_maximum;
  real step_scaling;
  real eps_g;
  real eps_f;
  real eps_x;
  int hessian_approximate_dimension;
  int max_iteration;
  LBFGSB_CUDA_MODE mode;
};

template <typename real>
struct LBFGSB_CUDA_SUMMARY {
  real residual_g;
  real residual_f;
  real residual_x;
  int num_iteration;
  int info;
};

template <typename real>
struct LBFGSB_CUDA_STATE {
  cublasContext* m_cublas_handle;
  std::function<int(real*, real&, real*, const cudaStream_t&,
                    const LBFGSB_CUDA_SUMMARY<real>&)>
      m_funcgrad_callback;
  std::function<int(real*, real&, real*, const cudaStream_t&,
                    const LBFGSB_CUDA_SUMMARY<real>&)>
      m_after_iteration_callback;
  std::function<int(real*, real&, real*, const cudaStream_t&,
                    const LBFGSB_CUDA_SUMMARY<real>&)>
      m_customized_stopping_callback;
};

namespace lbfgsbcuda {

/*************************************************************************
The  subroutine  minimizes  the  function  F(x) of N arguments with simple
constraints using a quasi-Newton method (LBFGS scheme) which is  optimized
to use a minimum amount of memory.

The subroutine generates the approximation of an inverse Hessian matrix by
using information about the last M steps of the algorithm (instead  of N).
It lessens a required amount of memory from a value  of  order  N^2  to  a
value of order 2*N*M.

This subroutine uses the FuncGrad subroutine which calculates the value of
the function F and gradient G in point X. The programmer should define the
FuncGrad subroutine by himself.  It should be noted  that  the  subroutine
doesn't need to waste  time for memory allocation of array G, because  the
memory is allocated in calling the  subroutine.  Setting  a  dimension  of
array G each time when calling a subroutine will excessively slow down  an
algorithm.

The programmer could also redefine the LBFGSNewIteration subroutine  which
is called on each new step. The current point X, the function value F  and
the gradient G are passed  into  this  subroutine.  It  is  reasonable  to
redefine the subroutine for better debugging, for  example,  to  visualize
the solution process.

Input parameters:
N       -   problem dimension. N>0
M       -   number of  corrections  in  the  BFGS  scheme  of  Hessian
approximation  update.  Recommended value:  3<=M<=7.   The
smaller value causes worse convergence,  the  bigger  will
not  cause  a  considerably  better  convergence, but will
cause a fall in the performance. M<=N.
X       -   initial solution approximation.
Array whose index ranges from 1 to N.
EpsG    -   positive number which defines a precision of  search.  The
subroutine finishes its work if the condition ||G|| < EpsG
is satisfied, where ||.|| means Euclidian norm, G - gradient
projection onto a feasible set, X - current approximation.
EpsF    -   positive number which defines a precision of  search.  The
subroutine  finishes  its  work if on iteration number k+1
the condition |F(k+1)-F(k)| <= EpsF*max{|F(k)|, |F(k+1)|, 1}
is satisfied.
EpsX    -   positive number which defines a precision of  search.  The
subroutine  finishes  its  work if on iteration number k+1
the condition |X(k+1)-X(k)| <= EpsX is satisfied.
MaxIts  -   maximum number of iterations.
If MaxIts=0, the number of iterations is unlimited.
NBD     -   constraint type. If NBD(i) is equal to:
* 0, X(i) has no constraints,
* 1, X(i) has only lower boundary,
* 2, X(i) has both lower and upper boundaries,
* 3, X(i) has only upper boundary,
Array whose index ranges from 1 to N.
L       -   lower boundaries of X(i) variables.
Array whose index ranges from 1 to N.
U       -   upper boundaries of X(i) variables.
Array whose index ranges from 1 to N.

Output parameters:
X       -   solution approximation.
Array whose index ranges from 1 to N.
Info    -   a return code:
* -2 unknown internal error,
* -1 wrong parameters were specified,
* 0 interrupted by user,
* 1 relative function decreasing is less or equal to EpsF,
* 2 step is less or equal to EpsX,
* 4 gradient norm is less or equal to EpsG,
* 5 number of iterations exceeds MaxIts.

FuncGrad routine description. User-defined.
Input parameters:
X   -   array whose index ranges from 1 to N.
Output parameters:
F   -   function value at X.
G   -   function gradient.
Array whose index ranges from 1 to N.
The memory for array G has already been allocated in the calling subroutine,
and it isn't necessary to allocate it in the FuncGrad subroutine.

NEOS, November 1994. (Latest revision June 1996.)
Optimization Technology Center.
Argonne National Laboratory and Northwestern University.

Written by Ciyou Zhu in collaboration with
R.H. Byrd, P. Lu-Chen and J. Nocedal.
*************************************************************************/
template <typename real>
LBFGSB_CUDA_FUNCTION void lbfgsbminimize(const int& n,
                                         const LBFGSB_CUDA_STATE<real>& state,
                                         const LBFGSB_CUDA_OPTION<real>& option,
                                         real* x, const int* nbd, const real* l,
                                         const real* u,
                                         LBFGSB_CUDA_SUMMARY<real>& summary);

template <typename real>
LBFGSB_CUDA_FUNCTION void lbfgsbdefaultoption(LBFGSB_CUDA_OPTION<real>& option);
}  // namespace lbfgsbcuda

#endif  // CULBFGSB_CULBFGSB_H_