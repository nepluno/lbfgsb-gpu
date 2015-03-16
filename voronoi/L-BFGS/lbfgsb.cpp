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


#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "lbfgsb.h"

inline void lbfgsbactive(const int& n,
	const real* l,
	const real* u,
	const int* nbd,
	real* x);
inline void lbfgsbbmv(const int& m,
	const real* sy,
	real* wt,
	const int& col,
	const int& iPitch,
	const real* v,
	real* p,
	const cudaStream_t& stream,
	int& info);
inline void lbfgsbcauchy(const int& n,
	const real* x,
	const real* l,
	const real* u,
	const int* nbd,
	const real* g,
	real* t,
	real* xcp,
	real* xcpb,
	const int& m,
	const real* wy,
	const real* ws,
	const real* sy,
	const int iPitch,
	real* wt,
	const real& theta,
	const int& col,
	const int& head,
	real* p,
	real* c,
	real* v,
	int& nint,
	const real& sbgnrm,
	real* buf_s_r,
	real* buf_array_p,
	const cudaStream_t* streamPool,
	int& info);
inline void lbfgsbcmprlb(const int& n,
     const int& m,
     const real* x,
     const real* g,
     const real* ws,
     const real* wy,
     const real* sy,
     real* wt,
     const real* z,
     real* r,
     real* wa,
     const real& theta,
     const int& col,
     const int& head,
     int& info,
     real* workvec,
     real* workvec2,
	 const int& iPitch,
	 const cudaStream_t& stream
	 );
inline void lbfgsbformk(const int& n,
	const int& iupdat,
	const bool& updatd,
	real* wn,
	real* wn1,
	const int& m,
	const real* ws,
	const real* wy,
	const real* sy,
	const real& theta,
	const int& col,
	const int& head,
	int& info,
	real* workvec,
	real* workmat,
	real* buf_array_p,
	const int& iPitch_wn,
	const int& iPitch_ws,
	const cudaStream_t* streamPool
	);
inline void lbfgsbformt(const int& m,
	real* wt,
	const real* sy,
	const real* ss,
	const int& col,
	const real& theta,
	int& info,
	const int& iPitch,
	const cudaStream_t* streamPool
	);
inline void lbfgsblnsrlb(const int& n,
	const real* l,
	const real* u,
	const int* nbd,
	real* x,
	const real& f,
	real& fold,
	real& gd,
	real& gdold,
	const real* g,
	real* d,
	real* r,
	real* t,
	const real* z,
	real& stp,
	real& dnrm,
	real& dtd,
	real& xstep,
	real& stpmx,
	const int& iter,
	int& ifun,
	int& iback,
	int& nfgv,
	int& info,
	int& task,
	int& csave,
	int* isave,
	real* dsave,
	real* buf_s_r,
	const cudaStream_t* streamPool
	);
inline void lbfgsbmatupdsub(const int& n,
	const int& m,
	real* wy,
	real* sy,
	const real* r,
	const real* d,
	int& itail,
	const int& iupdat,
	int& col,
	int& head,
	const real& dr,
	const int& iPitch0,
	const int& iPitch_i,
	const int& iPitch_j
	);
inline void lbfgsbmatupd(const int& n,
	const int& m,
	real* ws,
	real* wy,
	real* sy,
	real* ss,
	const real* d,
	const real* r,
	int& itail,
	const int& iupdat,
	int& col,
	int& head,
	real& theta,
	const real& rr,
	const real& dr,
	const real& stp,
	const real& dtd,
	const int& iPitch,
	real* buf_array_p,
	const cudaStream_t* streamPool
	);
inline void lbfgsbprojgr(const int& n,
	const real* l,
	const real* u,
	const int* nbd,
	const real* x,
	const real* g,
	real* buf_n,
	real* sbgnrm_h,
	real* sbgnrm_d,
	const cudaStream_t& stream
	);
inline void lbfgsbsubsm(const int& n,
	const int& m,
	const real* l,
	const real* u,
	const int* nbd,
	real* x,
	real* d,
	const real* ws,
	const real* wy,
	const real& theta,
	const int& col,
	const int& head,
	real* wv,
	real* wn,
	int& info,
	const int& iPitch_wn,
	const int& iPitch_ws,
	real* buf_array_p,
	real* buf_s_r,
	int* bufi_s_r,
	const cudaStream_t& stream
	);
inline void lbfgsbdcsrch(const real& f,
	const real& g,
	real& stp,
	const real& ftol,
	const real& gtol,
	const real& xtol,
	const real& stpmin,
	const real& stpmax,
	int& task,
	int* isave,
	real* dsave);
inline void lbfgsbdcstep(real& stx,
     real& fx,
     real& dx,
     real& sty,
     real& fy,
     real& dy,
     real& stp,
     const real& fp,
     const real& dp,
     bool& brackt,
     const real& stpmin,
     const real& stpmax);
inline bool lbfgsbdpofa(real* a, const int& n, const int& iPitch);
inline void lbfgsbdtrsl(real* t,
	const int& n,
	const int& iPitch,
	real* b,
	const int& job,
	int& info);
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

	Original Ver.:
    Optimization Technology Center.
    Argonne National Laboratory and Northwestern University.

    Written by Ciyou Zhu in collaboration with
    R.H. Byrd, P. Lu-Chen and J. Nocedal.

	
	Current Ver.:
	HKU, July 2012.
	Laboratory of Computer Graphics and Vision.
	The University of Hong Kong.
	
	Restructured by Yun Fei.
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
     int& info
	 )
{
    real f;
    real* g;
    real* xold;
    real* xdiff;
    real* z;
    real* zb;
    real* r;
    real* d;
    real* t;
    real* wa;
    real* buf_n_r;
	real* workvec;
	real* workvec2;
	
	real* workmat;
	real* ws;
	real* wy;
	real* sy;
	real* ss;
	real* yy;
	real* wt;
	real* wn;
	real* snd;
	real* buf_array_p;
	real* buf_array_p1;

    int* bufi_n_r;

    int csave;
    int task;
    bool updatd;
    bool wrk;
    int iback;
    int head;
    int col;
    int iter;
    int itail;
    int iupdat;
    int nint;
    int nfgv;
    int internalinfo;
    int ifun;
    real theta;
    real fold;
    real dr;
    real rr;
    real dnrm;
    real xstep;
    real sbgnrm; 
    real ddum;
    real dtd;
    real gd;
    real gdold;
    real stp;
    real stpmx;
    real tf;

	memAlloc<real>(&workvec, m);
	memAlloc<real>(&workvec2, 2 * m);
	memAlloc<real>(&g, n);
	memAlloc<real>(&xold, n);
	memAlloc<real>(&xdiff, n);
	memAlloc<real>(&z, n);
	memAlloc<real>(&zb, n);
	memAlloc<real>(&r, n);
	memAlloc<real>(&d, n);
	memAlloc<real>(&t, n);
	memAlloc<real>(&wa, 8 * m);
	memAlloc<real>(&buf_n_r, n);
	memAlloc<real>(&buf_array_p, m * n * 2);	
	memAlloc<real>(&buf_array_p1, m * n * 2);	

	size_t pitch0 = m;
	memAllocPitch<real>(&ws, m, n, &pitch0);
	memAllocPitch<real>(&wy, m, n, NULL);
	memAllocPitch<real>(&sy, m, m, NULL);
	memAllocPitch<real>(&ss, m, m, NULL);
	memAllocPitch<real>(&yy, m, m, NULL);
	memAllocPitch<real>(&wt, m, m, NULL);
	memAllocPitch<real>(&workmat, m, m, NULL);

	size_t pitch1 = m * 2;
	memAllocPitch<real>(&wn, m * 2, m * 2, &pitch1);
	memAllocPitch<real>(&snd, m * 2, m * 2, NULL);
	
	memAlloc<int>(&bufi_n_r, n);

	real* sbgnrm_h;
	real* sbgnrm_d;

	real* dsave13;
	memAllocHost<real>(&sbgnrm_h, &sbgnrm_d, sizeof(real));
	memAllocHost<real>(&dsave13, NULL, 16 * sizeof(real));

	int* isave2 = (int*)(dsave13 + 13);
	
	real epsx2 = epsx * epsx;

	cudaStream_t streamPool[16] = {NULL};

	const static int MAX_STREAM = 10;


	for(int i = 0; i < MAX_STREAM; i++)
		cutilSafeCall(cudaStreamCreate(streamPool + i));

    col = 0;
    head = 0; 
    theta = 1;
    iupdat = 0;
    updatd = false;
    iter = 0;
    nfgv = 0;
    nint = 0;
    internalinfo = 0;
	task = 0;

	lbfgsbcuda::CheckBuffer(x, n, n);
    lbfgsbactive(n, l, u, nbd, x);
	lbfgsbcuda::CheckBuffer(x, n, n);
	memCopyAsync(xold, x, n * sizeof(real), cudaMemcpyDeviceToDevice);
    funcgrad(x, f, g, NULL);

    nfgv = 1;
    lbfgsbprojgr(n, l, u, nbd, x, g, buf_n_r, sbgnrm_h, sbgnrm_d, NULL);
	cutilSafeCall(cudaThreadSynchronize());
	sbgnrm = *sbgnrm_h;
    if( sbgnrm<=epsg )
    {
        info = 4;
        return;
    }
    while(true)
    {
		//Streaming Start

		if( col!=0 && updatd )
		{
			lbfgsbformk(n, iupdat, updatd, wn, snd, m, ws, wy, sy,
					theta, col, head, internalinfo, workvec, workmat, buf_array_p, pitch1,
					pitch0, streamPool);
			if( internalinfo!=0 )
			{
				internalinfo = 0;
				col = 0;
				head = 0;
				theta = 1;
				iupdat = 0;
				updatd = false;
				continue;
			}
		}

		lbfgsbcauchy(n, x, l, u, nbd, g,
			t, z, zb, m, wy, ws, sy, pitch0, wt,
			theta, col, head, wa, wa + 2 * m, wa + 6 * m, nint, sbgnrm, buf_n_r, buf_array_p1, streamPool + 3, 
			internalinfo);
		if( internalinfo!=0 )
		{
			internalinfo = 0;
			col = 0;
			head = 0;
			theta = 1;
			iupdat = 0;
			updatd = false;
			continue;
		}


		cudaDeviceSynchronize();

        if( col!=0 )
        {
            lbfgsbcmprlb(n, m, x, g, ws, wy, sy,
				wt, zb, r, wa, theta, col, head, 
				internalinfo, workvec, workvec2, pitch0, streamPool[0]);
			if( internalinfo!=0 )
			{
				internalinfo = 0;
				col = 0;
				head = 0;
				theta = 1;
				iupdat = 0;
				updatd = false;
				continue;
			}

            lbfgsbsubsm(n, m, l, u, nbd, z, r, ws, wy,
				theta, col, head, wa, wn, internalinfo, 
				pitch1, pitch0, buf_array_p, buf_n_r, bufi_n_r, streamPool[0]);

            if( internalinfo!=0 )
            {
                internalinfo = 0;
                col = 0;
                head = 0;
                theta = 1;
                iupdat = 0;
                updatd = false;
                continue;
            }
        }
		lbfgsbcuda::minimize::vsub_v(n, z, x, d);
        task = 0;
        while(true)
        {
			lbfgsbcuda::CheckBuffer(x, n, n);
            lbfgsblnsrlb(n, l, u, nbd, x, f,
				fold, gd, gdold, g, d, r, t, z, 
				stp, dnrm, dtd, xstep, stpmx, iter, ifun, iback, nfgv, 
				internalinfo, task, csave, isave2, dsave13, buf_n_r, streamPool);
			lbfgsbcuda::CheckBuffer(x, n, n);
            if( internalinfo!=0||iback>=20||task!=1 )
            {
                break;
            }
            funcgrad(x, f, g, streamPool[1]);

       }
        iter = iter + 1;
		// finish debug

		lbfgsbprojgr(n, l, u, nbd, x, g, buf_n_r, sbgnrm_h, sbgnrm_d, streamPool[1]);

		lbfgsbcuda::minimize::vdiffxchg_v(n, xdiff, xold, x, streamPool[2]);
		lbfgsbcuda::CheckBuffer(xdiff, n, n);
		lbfgsbcuda::CheckBuffer(xold, n, n);
		lbfgsbcuda::minimize::vdot_vv(n, xdiff, xdiff, tf, streamPool[2]);

		lbfgsbcuda::minimize::vsub_v(n, g, r, r, streamPool[3]);
		lbfgsbcuda::minimize::vdot_vv(n, r, r, rr, streamPool[3]);

		ddum = fmaxf(fabs(fold), fmaxf(fabs(f), real(1)));

		if( fold - f <= epsf * ddum )
		{
			info = 1;
			break;
		}
		if( iter > maxits && maxits > 0 )
		{
			info = 5;
			break;
		}

		if( stp == 1 )
		{
			dr = gd-gdold;
			ddum = -gdold;
		}
		else
		{
			dr = (gd - gdold) * stp;
			ddum = -gdold * stp;
		}
		if( tf <= epsx2 )
		{
			info = 2;
			break;
		}
		sbgnrm = *sbgnrm_h;
		if( sbgnrm <= epsg )
		{
			info = 4;
			break;
		}
		if( dr <= machineepsilon * ddum )
		{
			updatd = false;
		}
		else
		{
			updatd = true;
			if(iupdat < m)
				iupdat++;

			lbfgsbmatupd(n, m, ws, wy, sy, ss, d, r, itail, 
				iupdat, col, head, theta, rr, dr, stp, dtd, pitch0,
				buf_array_p, streamPool);
			lbfgsbformt(m, wt, sy, ss, col, theta, internalinfo, pitch0, streamPool);
		}

		if( internalinfo!=0 )
		{
			memCopyAsync(x, t, n * sizeof(real), cudaMemcpyDeviceToDevice, streamPool[1]);
			memCopyAsync(g, r, n * sizeof(real), cudaMemcpyDeviceToDevice, streamPool[1]);
			f = fold;

			if( col==0 )
			{
				task = 2;
				iter = iter+1;
				info = -2;
				break;
			}
			else
			{
				internalinfo = 0;
				col = 0;
				head = 0;
				theta = 1;
				iupdat = 0;
				updatd = false;
				break;
			}
		}
    }


	for(int i = 0; i < MAX_STREAM; i++)
		cudaStreamDestroy(streamPool[i]);

	memFree(workvec);
	memFree(workvec2);
	memFree(g);
	memFree(xold);
	memFree(xdiff);
	memFree(z);
	memFree(zb);
	memFree(r);
	memFree(d);
	memFree(t);
	memFree(wa);
	memFree(buf_n_r);

	//pitch = m
	memFree(ws);
	memFree(wy);
	memFree(sy);
	memFree(ss);
	memFree(yy);
	memFree(wt);
	memFree(workmat);

	//pitch = 2 * m
	memFree(wn);
	memFree(snd);
	memFree(buf_array_p);	
	memFree(buf_array_p1);	

	memFree(bufi_n_r);
	memFreeHost(sbgnrm_h);
	memFreeHost(dsave13);

	printf("Iter: %d\n", iter);
}

inline void lbfgsbactive(const int& n,
	const real* l,
	const real* u,
	const int* nbd,
	real* x)
{

	lbfgsbcuda::active::prog0(n, l, u, nbd,
		x);
}

inline void lbfgsbbmv(const int& m,
     const real* sy,
     real* wt,
     const int& col,
	 const int& iPitch,
     const real* v,
     real* p,	
	 const cudaStream_t& stream,
     int& info)
{
	if( col==0 )
	{
		return;
	}

	lbfgsbcuda::bmv::prog0(sy, col, iPitch, v, p, stream);

	lbfgsbcuda::bmv::prog1(wt, col, iPitch, v, p, stream);

	lbfgsbcuda::bmv::prog2(sy, wt, col, iPitch, v, p, stream);
	info = 0;
}


inline void lbfgsbcauchy(const int& n,
     const real* x,
     const real* l,
     const real* u,
     const int* nbd,
     const real* g,
     real* t,
     real* xcp,
     real* xcpb,
     const int& m,
     const real* wy,
     const real* ws,
     const real* sy,
	 const int iPitch,
     real* wt,
     const real& theta,
     const int& col,
     const int& head,
     real* p,
     real* c,
     real* v,
     int& nint,
     const real& sbgnrm,
	 real* buf_s_r,
	 real* buf_array_p,
	 const cudaStream_t* streamPool,
     int& info)
{
	info = 0;
	lbfgsbcuda::CheckBuffer(p, m, 2 * m);
	lbfgsbcuda::CheckBuffer(c, m, 2 * m);
	lbfgsbcuda::CheckBuffer(v, m, 2 * m);
	lbfgsbcuda::cauchy::prog0(
		n, x, l, u, nbd, g, t, xcp, xcpb, m, wy, ws, sy,
		iPitch, wt, theta, col, head, p, c, v, nint, sbgnrm,
		buf_s_r, buf_array_p, streamPool);
	lbfgsbcuda::CheckBuffer(p, m, 2 * m);
	lbfgsbcuda::CheckBuffer(c, m, 2 * m);
	lbfgsbcuda::CheckBuffer(v, m, 2 * m);

	lbfgsbbmv(m, sy, wt, col, iPitch, c, p, streamPool[1], info);
}


inline void lbfgsbcmprlb(const int& n,
     const int& m,
     const real* x,
     const real* g,
     const real* ws,
     const real* wy,
     const real* sy,
     real* wt,
     const real* z,
     real* r,
     real* wa,
     const real& theta,
     const int& col,
     const int& head,
     int& info,
     real* workvec,
     real* workvec2,
	 const int& iPitch,
	 const cudaStream_t& stream
	 )
{

	lbfgsbcuda::CheckBuffer(wy, iPitch, iPitch * n);
	lbfgsbcuda::CheckBuffer(ws, iPitch, iPitch * n);
	lbfgsbcuda::cmprlb::prog1(
		n, col, head, m, iPitch, wa,
		wy, ws, theta, z, x, g, r, stream);
	lbfgsbcuda::CheckBuffer(r, n, n);
}

inline void lbfgsbformk(const int& n,
     const int& iupdat,
     const bool& updatd,
     real* wn,
     real* wn1,
     const int& m,
     const real* ws,
     const real* wy,
     const real* sy,
     const real& theta,
     const int& col,
     const int& head,
     int& info,
     real* workvec,
     real* workmat,
	 real* buf_array_p,
	 const int& iPitch_wn,
	 const int& iPitch_ws,
	 const cudaStream_t* streamPool
	 )
{
	int ipntr = head + col - 1;
	if( ipntr >= m )
	{
		ipntr = ipntr - m;
	}

	lbfgsbcuda::formk::prog1(
		n, ipntr, wn1, buf_array_p, wy, head, m,
		col, iPitch_ws, iPitch_wn, streamPool);

	lbfgsbcuda::formk::prog2(wn1, col, m, iPitch_wn, streamPool);

	lbfgsbcuda::CheckBuffer(wn1, iPitch_wn, iPitch_wn * m * 2);

	int jy = col - 1;
	int jpntr = head + col - 1;
	if( jpntr >= m )
	{
		jpntr = jpntr - m;
	}

	lbfgsbcuda::formk::prog3(jpntr, head, m, col, n, iPitch_ws, iPitch_wn,
		jy, ws, wy, buf_array_p, wn1, streamPool);
	lbfgsbcuda::CheckBuffer(wn1, iPitch_wn, iPitch_wn * m * 2);

	lbfgsbcuda::formk::prog4(col, iPitch_wn, iPitch_ws, m, wn1, theta, sy, wn, streamPool);
 	lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * m * 2);

	lbfgsbcuda::dpofa::prog0(wn, col, iPitch_wn, 0, streamPool[2]);
 	lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * m * 2);
// 
	lbfgsbcuda::formk::prog5(col, iPitch_wn, wn, streamPool);
 	lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * m * 2);

	lbfgsbcuda::dpofa::prog0(wn, col, iPitch_wn, col, streamPool[2]);
 	lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * m * 2);
}


inline void lbfgsbformt(const int& m,
     real* wt,
     const real* sy,
     const real* ss,
     const int& col,
     const real& theta,
     int& info,
	 const int& iPitch,
	 const cudaStream_t* streamPool
	 )
{
	lbfgsbcuda::CheckBuffer(wt, iPitch, col * iPitch);
	lbfgsbcuda::CheckBuffer(ss, iPitch, col * iPitch);
	lbfgsbcuda::CheckBuffer(sy, iPitch, col * iPitch);
	lbfgsbcuda::formt::prog01(col, sy, ss, wt, iPitch, theta, streamPool[0]);
	lbfgsbcuda::CheckBuffer(wt, iPitch, col * iPitch);
	lbfgsbcuda::dpofa::prog0(wt, col, iPitch, 0, streamPool[0]);
	lbfgsbcuda::CheckBuffer(wt, iPitch, col * iPitch);

	info = 0;
}

inline void lbfgsblnsrlb(const int& n,
     const real* l,
     const real* u,
     const int* nbd,
     real* x,
     const real& f,
     real& fold,
     real& gd,
     real& gdold,
     const real* g,
     real* d,
     real* r,
     real* t,
     const real* z,
     real& stp,
     real& dnrm,
     real& dtd,
     real& xstep,
     real& stpmx,
     const int& iter,
     int& ifun,
     int& iback,
     int& nfgv,
     int& info,
     int& task,
     int& csave,
     int* isave,
     real* dsave,
	 real* buf_s_r,
	 const cudaStream_t* streamPool
	 )
{
	int addinfo;

	addinfo = 0;
	const static real big = 1.0E10;
	const static real ftol = 1.0E-3;
	const static real gtol = 0.9E0;
	const static real xtol = 0.1E0;
	real* stpmx_host = NULL;
	real* stpmx_dev;


	if( task != 1 )
	{
		cudaHostAlloc(&stpmx_host, sizeof(real), cudaHostAllocMapped);
		cudaHostGetDevicePointer(&stpmx_dev, stpmx_host, 0);

		lbfgsbcuda::minimize::vdot_vv(n, d, d, dtd, streamPool[0]);

		*stpmx_host = stpmx = big;

		if( iter == 0 )
		{
			stpmx = 1;
		}
		else
		{
			lbfgsbcuda::lnsrlb::prog0(n, d, nbd, u, x, l, buf_s_r, stpmx_host, stpmx_dev, streamPool[1]);

		}


		memCopyAsync(t, x, n * sizeof(real), cudaMemcpyDeviceToDevice, streamPool[2]);
		memCopyAsync(r, g, n * sizeof(real), cudaMemcpyDeviceToDevice, streamPool[3]);
		fold = f;
		ifun = 0;
		iback = 0;
		csave = 0;
	}
	lbfgsbcuda::minimize::vdot_vv(n, g, d, gd, streamPool[4]);

	cudaDeviceSynchronize();

	if( task != 1 )
	{
		if(iter != 0) {
			stpmx = *stpmx_host;
			if(stpmx == big)
				stpmx = 1;
		}
		dnrm = sqrt(dtd);
		if( iter==0 )
		{
			stp = fmaxf(1.0, fminf(1.0 / dnrm, stpmx));
		}
		else
		{
			stp = 1;
		}
		cudaFreeHost(stpmx_host);
	}
	if( ifun == 0 )
	{
		gdold = gd;
		if( gd >= 0 )
		{
			info = -4;
			return;
		}
	}
	lbfgsbdcsrch(f, gd, stp, ftol, gtol, xtol, machineepsilon, stpmx, csave, isave, dsave);
	if( csave!=3 )
	{
		task = 1;
		ifun = ifun+1;
		nfgv = nfgv+1;
		iback = ifun-1;
		if( stp==1 )
		{
			memCopyAsync(x, z, n * sizeof(real), cudaMemcpyDeviceToDevice, streamPool[1]);
		}
		else
		{
			lbfgsbcuda::lnsrlb::prog2(n, x, d, t, stp, streamPool[1]);
		}
	}
	else
	{
		task = 5;
	}

	xstep = stp * dnrm;
}


void lbfgsbmatupdsub(const int& n,
	const int& m,
	real* wy,
	real* sy,
	const real* r,
	const real* d,
	int& itail,
	const int& iupdat,
	int& col,
	int& head,
	const real& dr,
	const int& iPitch0,
	const int& iPitch_i,
	const int& iPitch_j
	) 
{
	vmove_mv(wy, r, itail, 0, n - 1, 0, iPitch0);	
	for(int j = 0; j < col-1; j++)
	{
		int pointr = Modular((head + j), m);
		sy[(col - 1) * iPitch_i + j * iPitch_j] = vdot_vm(d, wy, 0, n - 1, pointr, 0, iPitch0);
	}
	sy[(col - 1) * iPitch0 + col - 1] = dr;
}

inline void lbfgsbmatupd(const int& n,
	const int& m,
	real* ws,
	real* wy,
	real* sy,
	real* ss,
	const real* d,
	const real* r,
	int& itail,
	const int& iupdat,
	int& col,
	int& head,
	real& theta,
	const real& rr,
	const real& dr,
	const real& stp,
	const real& dtd,
	const int& iPitch,
	real* buf_array_p,
	const cudaStream_t* streamPool
	)
{
	col = iupdat;
	itail = Modular((head + iupdat - 1), m);

	theta = rr / dr;

	lbfgsbcuda::matupd::prog0(
		n, m, wy, sy, r, d, itail, iupdat,
		col, head, dr, iPitch, iPitch, 1, buf_array_p, streamPool[1]		
	);

	lbfgsbcuda::matupd::prog0(
		n, m, ws, ss, d, d, itail, iupdat,
		col, head, stp * stp * dtd, iPitch,
		1, iPitch, buf_array_p + n / 2, streamPool[2]
	);
}


inline void lbfgsbprojgr(const int& n,
     const real* l,
     const real* u,
     const int* nbd,
     const real* x,
     const real* g,
	 real* buf_n,
	 real* sbgnrm_h,
	 real* sbgnrm_d,
	 const cudaStream_t& stream
	 )
{
	lbfgsbcuda::projgr::prog0(n, l, u, nbd, x, g, buf_n, sbgnrm_h, sbgnrm_d, stream);
}


inline void lbfgsbsubsm(const int& n,
     const int& m,
     const real* l,
     const real* u,
     const int* nbd,
     real* x,
     real* d,
     const real* ws,
     const real* wy,
     const real& theta,
     const int& col,
     const int& head,
     real* wv,
     real* wn,
     int& info,
	 const int& iPitch_wn,
	 const int& iPitch_ws,
	 real* buf_array_p,
	 real* buf_s_r,
	 int* bufi_s_r,
	 const cudaStream_t& stream
	 )
{

	lbfgsbcuda::subsm::prog0(n, head, m, col, iPitch_ws, 
		buf_array_p, wy, ws, d, wv, theta, stream);

	lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
	lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * m);
	lbfgsbcuda::subsm::prog1(wn, col, iPitch_wn, wv, stream);
	lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);

	lbfgsbcuda::CheckBuffer(d, n, n);
	lbfgsbcuda::subsm::prog2(n, col, head, m, iPitch_ws, wv, wy, ws, theta, d, stream);
	lbfgsbcuda::CheckBuffer(d, n, n);

	lbfgsbcuda::subsm::prog3(n, d, nbd, buf_s_r, bufi_s_r, x, u, l, stream);
	lbfgsbcuda::CheckBuffer(x, n, n);
	lbfgsbcuda::CheckBuffer(d, n, n);
}


inline void lbfgsbdcsrch(const real& f,
     const real& g,
     real& stp,
     const real& ftol,
     const real& gtol,
     const real& xtol,
     const real& stpmin,
     const real& stpmax,
     int& task,
     int* isave,
     real* dsave)
{
    register bool brackt;
    register int stage;
    register real finit;
    register real ftest;
    register real fm;
    register real fx;
    register real fxm;
    register real fy;
    register real fym;
    register real ginit;
    register real gtest;
    register real gm;
    register real gx;
    register real gxm;
    register real gy;
    register real gym;
    register real stx;
    register real sty;
    register real stmin;
    register real stmax;
    register real width;
    register real width1;

    const static real xtrapl = 1.1E0;
    const static real xtrapu = 4.0E0;
	int counter = 0;
    while(true)
    {
		counter++;
        if( task==0 )
        {
            if( stp < stpmin || stp > stpmax || g >= 0 || ftol < 0 || gtol < 0 || xtol < 0 || stpmin < 0 || stpmax < stpmin )
            {
                task = 2;
                return;
            }
            brackt = false;
            stage = 1;
            finit = f;
            ginit = g;
            gtest = ftol * ginit;
            width = stpmax - stpmin;
            width1 = width * 2.0;
            stx = 0;
            fx = finit;
            gx = ginit;
            sty = 0;
            fy = finit;
            gy = ginit;
            stmin = 0;
            stmax = stp+xtrapu*stp;
            task = 1;
            break;
        }
        else
        {
            brackt = (isave[0] == 1);
            stage = isave[1];
            ginit = dsave[0];
            gtest = dsave[1];
            gx = dsave[2];
            gy = dsave[3];
            finit = dsave[4];
            fx = dsave[5];
            fy = dsave[6];
            stx = dsave[7];
            sty = dsave[8];
            stmin = dsave[9];
            stmax = dsave[10];
            width = dsave[11];
            width1 = dsave[12];
        }
        ftest = finit + stp * gtest;
        if( stage == 1 && f <= ftest && g >= 0 )
        {
            stage = 2;
        }
        if( (brackt && (stp <= stmin || stp >= stmax || stmax - stmin <= xtol * stmax ) )
			|| (stp == stpmax && f <= ftest && g <= gtest) 
			|| (stp == stpmin && ( f > ftest || g >= gtest ))
			|| (f <= ftest && fabs(g) <= gtol * (-ginit)))
        {
            task = 3;
			break;
        }
        if( stage == 1 && f <= fx && f > ftest )
        {
            fm = f - stp * gtest;
            fxm = fx - stx * gtest;
            fym = fy - sty * gtest;
            gm = g - gtest;
            gxm = gx - gtest;
            gym = gy - gtest;
            lbfgsbdcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);
            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        }
        else
        {
            lbfgsbdcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
        }
        if( brackt )
        {
            if( fabs(sty - stx) >= 0.666666666666666667 * width1 )
            {
                stp = stx + 0.5 * (sty - stx);
            }
            width1 = width;
            width = fabs(sty - stx);
			stmin = fminf(stx, sty);
			stmax = fmaxf(stx, sty);
        }
        else
        {
            stmin = stp + xtrapl * (stp - stx);
            stmax = stp + xtrapu * (stp - stx);
        }
		stp *= stpscal;
        stp = fmaxf(stp, stpmin);
        stp = fminf(stp, stpmax);
        if( brackt && (stp <= stmin || stp >= stmax) || brackt && stmax - stmin <= xtol * stmax )
        {
            stp = stx;
        }
        task = 1;
        break;
    }
    if( brackt )
    {
        isave[0] = 1;
    }
    else
    {
        isave[0] = 0;
    }
    isave[1] = stage;
    dsave[0] = ginit;
    dsave[1] = gtest;
    dsave[2] = gx;
    dsave[3] = gy;
    dsave[4] = finit;
    dsave[5] = fx;
    dsave[6] = fy;
    dsave[7] = stx;
    dsave[8] = sty;
    dsave[9] = stmin;
    dsave[10] = stmax;
    dsave[11] = width;
    dsave[12] = width1;
}


inline void lbfgsbdcstep(real& stx,
     real& fx,
     real& dx,
     real& sty,
     real& fy,
     real& dy,
     real& stp,
     const real& fp,
     const real& dp,
     bool& brackt,
     const real& stpmin,
     const real& stpmax)
{
    register real gamma;
    register real p;
    register real q;
    register real r;
    register real s;
    register real sgnd;
    register real stpc;
    register real stpf;
    register real stpq;
    register real theta;
	register real stpstx;

	sgnd = dp * dx / fabs(dx);
	if( fp > fx )
	{
		theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
		s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
		gamma = s * sqrt((theta * theta) / (s * s) - dx / s * (dp / s));
		if( stp < stx )
		{
			gamma = -gamma;
		}
		p = gamma - dx + theta;
		q = gamma - dx + gamma + dp;
		r = p / q;
		stpstx = stp - stx;
		stpc = stx + r * stpstx;
		stpq = stx + dx / ((fx - fp) / stpstx + dx) * 0.5 * stpstx;
		if( fabs(stpc - stx) < fabs(stpq - stx) )
		{
			stpf = stpc;
		}
		else
		{
			stpf = stpc + (stpq - stpc) * 0.5;
		}
		brackt = true;
	}
	else if( sgnd < 0 )
	{
		theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
		s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
		gamma = s * sqrt((theta * theta) / (s * s) - dx * dp / (s * s));
		if( stp > stx )
		{
			gamma = -gamma;
		}
		p = gamma - dp + theta;
		q = gamma * 2.0 - dp + dx;
		r = p / q;
		stpstx = stx - stp;
		stpc = stp + r * stpstx;
		stpq = stp + dp / (dp - dx) * stpstx;
		if( fabs(stpc - stp) > fabs(stpq - stp) )
		{
			stpf = stpc;
		}
		else
		{
			stpf = stpq;
		}
		brackt = true;
	}
	else if( fabs(dp) < fabs(dx) )
	{
		theta = 3.0 * (fx-fp) / (stp - stx) + dx + dp;
		s = fmaxf(fabs(theta), fmaxf(fabs(dx), fabs(dp)));
		gamma = s * sqrt(fmaxf(0.0, (theta * theta) / (s * s) - dx * dp / (s * s)));
		if( stp>stx )
		{
			gamma = -gamma;
		}
		p = gamma - dp + theta;
		q = gamma + (dx - dp) + gamma;
		r = p / q;
		if( r < 0.0 && gamma != 0.0 )
		{
			stpc = stp + r * (stx-stp);
		}
		else if( stp > stx )
		{
			stpc = stpmax;
		}
		else
		{
			stpc = stpmin;
		}
		stpq = stp + dp / (dp - dx) * (stx - stp);
		if( fabs(stpc - stp) < fabs(stpq - stp) )
		{
			stpf = stpc;
		}
		else
		{
			stpf = stpq;
		}
		if( brackt )
		{
			stpf = fmaxf(stp + 0.666666666666667 * (sty - stp), stpf);
		}
		else
		{
			stpf = fmaxf(stpmax, stpf);
		}
	}
	else if( brackt )
	{
		theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
		s = fmaxf(fabs(theta), fmaxf(fabs(dy), fabs(dp)));
		gamma = s * sqrt((theta * theta) / (s * s) - dy / s * (dp / s));
		if( stp > sty )
		{
			gamma = -gamma;
		}
		p = gamma - dp + theta;
		q = gamma - dp + gamma + dy;
		r = p / q;
		stpc = stp + r * (sty - stp);
		stpf = stpc;
	}
	else if( stp > stx )
	{
		stpf = stpmax;
	}
	else
	{
		stpf = stpmin;
	}

	if( fp > fx )
	{
		sty = stp;
		fy = fp;
		dy = dp;
	}
	else
	{
		if( sgnd < 0 )
		{
			sty = stx;
			fy = fx;
			dy = dx;
		}
		stx = stp;
		fx = fp;
		dx = dp;
	}
	stp = stpf;
}

inline bool lbfgsbdpofa(real* a, const int& n, const int& iPitch)
{
    bool result;
    real s;
    real v;
    int j;
    int k;

    for(j = 0; j < n; j++)
    {
        s = 0.0;
        if( j >= 1 )
        {
            for(k = 0; k <= j - 1; k++)
            {
				v = vdot_mm(a, a, k, 0, k - 1, iPitch, j, 0, iPitch);
                a[k * iPitch + j] = (a[k * iPitch + j] - v) / a[k * iPitch + k];
                s = s + a[k * iPitch + j] * a[k * iPitch + j];
            }
        }
        s = a[j * iPitch + j]-s;
        if( s<=0.0 )
        {
            result = false;
            return result;
        }
        a[j * iPitch + j] = sqrt(s);
    }
    result = true;
    return result;
}


inline void lbfgsbdtrsl(real* t,
     const int& n,
	 const int& iPitch,
     real* b,
     const int& job,
     int& info)
{
    real temp;
    real v;
    int cse;
    int j;
    int jj;

    for(j = 0; j < n; j++)
    {
        if( t[j * iPitch + j] == 0.0 )
        {
            info = j;
            return;
        }
    }
    info = 0;
    cse = 1;
    if( job % 10 != 0 )
    {
        cse = 2;
    }
    if( job % 100/10 != 0 )
    {
        cse = cse+2;
    }
    if( cse == 1 )
    {
		b[0] = b[0] / t[0];
        if( n < 2 )
        {
            return;
        }
        for(j = 1; j < n; j++)
        {
            temp = -b[j - 1];
			vadd_vm(b, t, j, n - 1, j - 1, j, iPitch, temp);
            b[j] = b[j] / t[j * iPitch + j];
        }
        return;
    }
    if( cse==2 )
    {
        b[n - 1] = b[n - 1] / t[(n - 1) * iPitch + n - 1];
        if( n < 2 )
        {
            return;
        }
        for(j = n - 2; j >= 0; j--)
        {
            temp = -b[j + 1];
			vadd_vm(b, t, 0, j, j + 1, 0, iPitch, temp);
            b[j] = b[j] / t[j * iPitch + j];
        }
        return;
    }
    if( cse==3 )
    {
        b[n - 1] = b[n - 1] / t[(n - 1) * iPitch + n - 1];
        if( n<2 )
        {
            return;
        }
        for(jj = 2; jj <= n; jj++)
        {
            j = n - jj;
			v = vdot_vm(b, t, j + 1, j + jj - 1, j, j + 1, iPitch);
            b[j] = b[j] - v;
            b[j] = b[j] / t[j * iPitch + j];
        }
        return;
    }
    if( cse==4 )
    {
        b[0] = b[0] / t[0];
        if( n<2 )
        {
            return;
        }
        for(j = 1; j < n; j++)
        {
			v = vdot_vm(b, t, 0, j - 1, j, 0, iPitch);
            b[j] = b[j] - v;
            b[j] = b[j] / t[j * iPitch + j];
        }
        return;
    }
}
