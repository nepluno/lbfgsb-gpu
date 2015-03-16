/*************************************************************************
GPU Version:
Tsinghua University, Aug. 2012.

Written by Yun Fei in collaboration with
W. Wang and B. Wang

Original:
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

#ifndef _lbfgsb_h
#define _lbfgsb_h

#include "lbfgsbcuda.h"
#include "cutil_inline.h"
#include <cublas_v2.h>

extern int numIter;
extern bool bNewIteration;
extern FILE* f_result;
extern int nFuncCall;
extern bool bShowTestResults;
extern real stpscal;

inline int iAlignUp(int a, int b)
{
	return (a % b != 0) ?  (a - a % b + b) : a;
}

template<class T>
inline int memAlloc(T** ptr, size_t length) {
	void* ptr_;
	cudaMalloc(&ptr_, length * sizeof(T));
	*ptr = (T*)ptr_;
	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

template<class T>
inline int memAllocHost(T** ptr, T** ptr_dev, size_t length) {
	void* ptr_;
	cudaHostAlloc(&ptr_, length * sizeof(T), cudaHostAllocMapped);
	if(ptr_dev)
		cudaHostGetDevicePointer((void**)ptr_dev, ptr_, 0);
	*ptr = (T*)ptr_;

	memset(*ptr, 0, length);

	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

template<class T>
inline int memAllocPitch(T** ptr, size_t lx, size_t ly, size_t* pitch) {
	int bl = lx * sizeof(T);
	void* ptr_;
	bl = iAlignUp(bl, 32);
	if(pitch)
		*pitch = bl / sizeof(T);
	int length = bl * ly;
	cudaMalloc(&ptr_, length);
	*ptr = (T*)ptr_;

	cudaMemset(*ptr, 0, length);

	if(ptr_ != NULL)
		return 0;
	else
		return -1;
}

inline void memFree(void* ptr) {
	cudaFree(ptr);
}

inline void memFreeHost(void* ptr) {
	cudaFreeHost(ptr);
}

inline void memCopy(void* p1, const void* p2, size_t length, int type = 0) {
	//memcpy(p1, p2, length);
	cudaMemcpy(p1, p2, length, (cudaMemcpyKind)type);
}

inline void memCopyAsync(void* p1, const void* p2, size_t length, int type = 0, const cudaStream_t stream = NULL) {
	//memcpy(p1, p2, length);
	cudaMemcpyAsync(p1, p2, length, (cudaMemcpyKind)type, stream);
}

inline void memSet(void* p1, int _val, size_t length) {
	memset(p1, _val, length);
}

inline int fmaxf(int a, int b) {
	return a > b ? a : b;
}

inline int fminf(int a, int b) {
	return a > b ? b : a;
}

inline real fmaxf(real a, real b) {
	return a > b ? a : b;
}

inline real fminf(real a, real b) {
	return a > b ? b : a;
}

inline void vmove_vm(real* dst, const real* src, int ds, int de, int sr, int scs, int spitch) {
	int n = de - ds + 1;
	for(int i = 0; i < n; i++) {
		dst[ds + i] = src[(scs + i) * spitch + sr];
	}
}

inline void vmove_mv(real* dst, const real* src, int dr, int dcs, int dce, int ss, int dpitch) {
	int n = dce - dcs + 1;
	for(int i = 0; i < n; i++) {
		dst[(dcs + i) * dpitch + dr] = src[ss + i];
	}
}

inline void vadd_vm(real* dst, const real* src, int ds, int de, int sr, int scs, int spitch, real& alpha) 
{
	int n = de - ds + 1;
	for(int i = 0; i < n; i++) {
		dst[ds + i] += src[(scs + i) * spitch + sr] * alpha;
	}
}

inline void vmul_v(real* dst, int ds, int de, const real& alpha) 
{
	int n = de - ds + 1;
	for(int i = 0; i < n; i++) {
		dst[ds + i] *= alpha;
	}
}

inline void vadd_vv(real* dst, const real* src, int ds, int de, int sr, real& alpha) 
{
	vadd_vm(dst, src, ds, de, sr, 0, 1, alpha);
}

inline real vdot_vm(const real* dst, const real* src, int ds, int de, int sr, int scs, int spitch) 
{
	int n = de - ds + 1;
	real r = 0;
	for(int i = 0; i < n; i++) {
		r += dst[ds + i] * src[(scs + i) * spitch + sr];
	}
	return r;
}

inline real vdot_mm(const real* dst, const real* src, int dr, int dcs, int dce, int dpitch, int sr, int scs, int spitch) 
{
	int n = dce - dcs + 1;
	real r = 0;
	for(int i = 0; i < n; i++) {
		r += dst[(dcs + i) * dpitch + dr] * src[(scs + i) * spitch + sr];
	}
	return r;
}

inline real vdot_vv(const real* dst, const real* src, int ds, int de, int sr) 
{
	return vdot_vm(dst, src, ds, de, sr, 0, 1);
}

void funcgrad(real* x, real& f, real* g, const cudaStream_t& stream);

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
void lbfgsbminimize(const int& n,
	const int& m,
	real* x,
	const real& epsg,
	const real& epsf,
	const real& epsx,
	const int& maxits,
	const int* nbd,
	const real* l,
	const real* u,
	int& info);

#endif
