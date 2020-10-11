/*************************************************************************
NEOS, November 1994. (Latest revision June 1996.)
Optimization Technology Center.
Argonne National Laboratory and Northwestern University.

Written by Ciyou Zhu in collaboration with
R.H. Byrd, P. Lu-Chen and J. Nocedal.

Contributors:
    * Sergey Bochkanov (ALGLIB project). Translation from FORTRAN to
      pseudocode.

This software is freely available, but we  expect  that  all  publications
describing  work using this software, or all commercial products using it,
quote at least one of the references given below:
    * R. H. Byrd, P. Lu and J. Nocedal.  A Limited  Memory  Algorithm  for
      Bound Constrained Optimization, (1995), SIAM Journal  on  Scientific
      and Statistical Computing , 16, 5, pp. 1190-1208.
    * C. Zhu, R.H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for  large  scale  bound  constrained  optimization
      (1997), ACM Transactions on Mathematical Software,  Vol 23,  Num. 4,
      pp. 550 - 560.
*************************************************************************/

#ifndef CULBFGSB_LBFGSBCPU_H_
#define CULBFGSB_LBFGSBCPU_H_

#include "culbfgsb/ap.h"
#include "culbfgsb/lbfgsbcuda.h"

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
namespace lbfgsbcuda {
namespace cpu {
template <typename real>
void lbfgsbactive(const int& n, const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd,
                  ap::template_1d_array<real, true>& x,
                  ap::integer_1d_array& iwhere, bool& prjctd, bool& cnstnd,
                  bool& boxed);

template <typename real>
void lbfgsbbmv(const int& m, const ap::template_1d_array<real, true>& sy,
               ap::template_2d_array<real, true>& wt, const int& col,
               const ap::template_1d_array<real, true>& v,
               ap::template_1d_array<real, true>& p, int& info,
               ap::template_1d_array<real, true>& workvec);

template <typename real>
void lbfgsbcauchy(
    const int& n, const ap::template_1d_array<real, true>& x,
    const ap::template_1d_array<real, true>& l,
    const ap::template_1d_array<real, true>& u, const ap::integer_1d_array& nbd,
    const ap::template_1d_array<real, true>& g, ap::integer_1d_array& iorder,
    ap::integer_1d_array& iwhere, ap::template_1d_array<real, true>& t,
    ap::template_1d_array<real, true>& d,
    ap::template_1d_array<real, true>& xcp, const int& m,
    const ap::template_2d_array<real, true>& wy,
    const ap::template_2d_array<real, true>& ws,
    const ap::template_2d_array<real, true>& sy,
    ap::template_2d_array<real, true>& wt, const real& theta, const int& col,
    const int& head, ap::template_1d_array<real, true>& p,
    ap::template_1d_array<real, true>& c,
    ap::template_1d_array<real, true>& wbp,
    ap::template_1d_array<real, true>& v, int& nint,
    const ap::template_1d_array<real, true>& sg,
    const ap::template_1d_array<real, true>& yg, const real& sbgnrm, int& info,
    ap::template_1d_array<real, true>& workvec);

template <typename real>
void lbfgsbcmprlb(const int& n, const int& m,
                  const ap::template_1d_array<real, true>& x,
                  const ap::template_1d_array<real, true>& g,
                  const ap::template_2d_array<real, true>& ws,
                  const ap::template_2d_array<real, true>& wy,
                  const ap::template_2d_array<real, true>& sy,
                  ap::template_2d_array<real, true>& wt,
                  const ap::template_1d_array<real, true>& z,
                  ap::template_1d_array<real, true>& r,
                  ap::template_1d_array<real, true>& wa,
                  const ap::integer_1d_array& index, const real& theta,
                  const int& col, const int& head, const int& nfree,
                  const bool& cnstnd, int& info,
                  ap::template_1d_array<real, true>& workvec,
                  ap::template_1d_array<real, true>& workvec2);

template <typename real>
void lbfgsberrclb(const int& n, const int& m, const real& factr,
                  const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd, int& task, int& info,
                  int& k);

template <typename real>
void lbfgsbformk(const int& n, const int& nsub, const ap::integer_1d_array& ind,
                 const int& nenter, const int& ileave,
                 const ap::integer_1d_array& indx2, const int& iupdat,
                 const bool& updatd, ap::template_2d_array<real, true>& wn,
                 ap::template_2d_array<real, true>& wn1, const int& m,
                 const ap::template_2d_array<real, true>& ws,
                 const ap::template_2d_array<real, true>& wy,
                 const ap::template_2d_array<real, true>& sy, const real& theta,
                 const int& col, const int& head, int& info,
                 ap::template_1d_array<real, true>& workvec,
                 ap::template_2d_array<real, true>& workmat);

template <typename real>
void lbfgsbformt(const int& m, ap::template_2d_array<real, true>& wt,
                 const ap::template_2d_array<real, true>& sy,
                 const ap::template_2d_array<real, true>& ss, const int& col,
                 const real& theta, int& info);

void lbfgsbfreev(const int& n, int& nfree, ap::integer_1d_array& index,
                 int& nenter, int& ileave, ap::integer_1d_array& indx2,
                 const ap::integer_1d_array& iwhere, bool& wrk,
                 const bool& updatd, const bool& cnstnd, const int& iter);

template <typename real>
void lbfgsbhpsolb(const int& n, ap::template_1d_array<real, true>& t,
                  ap::integer_1d_array& iorder, const int& iheap);

template <typename real>
void lbfgsblnsrlb(
    const int& n, const ap::template_1d_array<real, true>& l,
    const ap::template_1d_array<real, true>& u, const ap::integer_1d_array& nbd,
    ap::template_1d_array<real, true>& x, const real& f, real& fold, real& gd,
    real& gdold, const ap::template_1d_array<real, true>& g,
    const ap::template_1d_array<real, true>& d,
    ap::template_1d_array<real, true>& r, ap::template_1d_array<real, true>& t,
    const ap::template_1d_array<real, true>& z, real& stp, real& dnrm,
    real& dtd, real& xstep, real& stpmx, const int& iter, int& ifun, int& iback,
    int& nfgv, int& info, int& task, const bool& boxed, const bool& cnstnd,
    int& csave, ap::integer_1d_array& isave,
    ap::template_1d_array<real, true>& dsave);

template <typename real>
void lbfgsbmatupd(const int& n, const int& m,
                  ap::template_2d_array<real, true>& ws,
                  ap::template_2d_array<real, true>& wy,
                  ap::template_2d_array<real, true>& sy,
                  ap::template_2d_array<real, true>& ss,
                  const ap::template_1d_array<real, true>& d,
                  const ap::template_1d_array<real, true>& r, int& itail,
                  const int& iupdat, int& col, int& head, real& theta,
                  const real& rr, const real& dr, const real& stp,
                  const real& dtd);

template <typename real>
void lbfgsbprojgr(const int& n, const ap::template_1d_array<real, true>& l,
                  const ap::template_1d_array<real, true>& u,
                  const ap::integer_1d_array& nbd,
                  const ap::template_1d_array<real, true>& x,
                  const ap::template_1d_array<real, true>& g, real& sbgnrm);

template <typename real>
void lbfgsbsubsm(const int& n, const int& m, const int& nsub,
                 const ap::integer_1d_array& ind,
                 const ap::template_1d_array<real, true>& l,
                 const ap::template_1d_array<real, true>& u,
                 const ap::integer_1d_array& nbd,
                 ap::template_1d_array<real, true>& x,
                 ap::template_1d_array<real, true>& d,
                 const ap::template_2d_array<real, true>& ws,
                 const ap::template_2d_array<real, true>& wy, const real& theta,
                 const int& col, const int& head, int& iword,
                 ap::template_1d_array<real, true>& wv,
                 ap::template_2d_array<real, true>& wn, int& info);

template <typename real>
void lbfgsbdcsrch(const real& f, const real& g, real& stp, const real& ftol,
                  const real& gtol, const real& xtol, const real& stpmin,
                  const real& stpmax, int& task, ap::integer_1d_array& isave,
                  ap::template_1d_array<real, true>& dsave, int& addinfo);

template <typename real>
void lbfgsbdcstep(real& stx, real& fx, real& dx, real& sty, real& fy, real& dy,
                  real& stp, const real& fp, const real& dp, bool& brackt,
                  const real& stpmin, const real& stpmax);

template <typename real>
bool lbfgsbdpofa(ap::template_2d_array<real, true>& a, const int& n);

template <typename real>
void lbfgsbdtrsl(ap::template_2d_array<real, true>& t, const int& n,
                 ap::template_1d_array<real, true>& b, const int& job,
                 int& info);

template <typename real>
void lbfgsbminimize(const int& n, const LBFGSB_CUDA_STATE<real>& state,
                    const LBFGSB_CUDA_OPTION<real>& option, real* x,
                    const int* nbd, const real* l, const real* u,
                    LBFGSB_CUDA_SUMMARY<real>& summary);
};  // namespace cpu
};  // namespace lbfgsbcuda

#endif  // CULBFGSB_LBFGSBCPU_H_
