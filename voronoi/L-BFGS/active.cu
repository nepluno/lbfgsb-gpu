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

#include "lbfgsbcuda.h"

namespace lbfgsbcuda {
	namespace active {
		__global__
		void kernel0(
			const int n,
			const real* l,
			const real* u,
			const int* nbd,
			real* x)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;

			int nbdi = nbd[i];
			real xi = x[i];
			real li = l[i];
			real ui = u[i];

			if(nbdi > 0) {
				if( nbdi <= 2 )
				{
					xi = maxr(xi, li);
				}
				else
				{
					xi = minr(xi, ui);
				}
			} 
			x[i] = xi;			
		}


		void prog0(
			const int& n,
			const real* l,
			const real* u,
			const int* nbd,
			real* x
			) 
		{

			kernel0<<<dim3(iDivUp(n, 512)), dim3(512)>>>
				(n, l, u, nbd, x);

		}
	};
};