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
	namespace cmprlb {
		__global__
		void kernel0(
		int n,
		real* r,
		const real* g)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			if(i >= n)
				return;

			r[i] = -g[i];
		}

		template<int bsize>
		__global__
		void kernel1(
		int nfree,
		const int col,
		const int head,
		const int m,
		const int iPitch,
		const real* wa,
		const real* wy,
		const real* ws,
		const real theta,
		const real* z,
		const real* x,
		const real* g,
		real* r
		)
		{
			const int i = blockIdx.x * blockDim.y + threadIdx.y;
			const int tidx = threadIdx.x; //8
			const int tidy = threadIdx.y; //64
			
			volatile __shared__ real sdata[(512 / bsize)][bsize+1];

			__shared__ real a[2][bsize+1];

			real mySum;

			if(tidy == 0 && tidx < col) {
				a[0][tidx] = wa[tidx];
				a[1][tidx] = theta * wa[col + tidx];
			}

			if(i < nfree && tidx < col) {
				const int pointr = Modular((head + tidx), m);
				__syncthreads();
				mySum = wy[i * iPitch + pointr] * a[0][tidx] + ws[i * iPitch + pointr] * a[1][tidx];
			} else
				mySum = 0;
			
			if(bsize > 1) {
				volatile real* smem = sdata[tidy] + tidx;
				*smem = mySum;

				__syncthreads();

				if(bsize > 4) {*smem = mySum = mySum + smem[4];}
				if(bsize > 2) {*smem = mySum = mySum + smem[2];}
				if(bsize > 1) {*smem = mySum = mySum + smem[1];}
			}

			if(tidx == 0 && i < nfree) {
				r[i] = -theta * (z[i] - x[i]) - g[i] + mySum;
			}
		}

		void prog0(
			const int n,
			real* r,
			const real* g,
			const cudaStream_t& stream
			)
		{
			kernel0<<<dim3(iDivUp(n, 512)), dim3(512), 0, stream>>>
				(n, r, g);
		}

		void prog1(
			const int nfree,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const real* wa,
			const real* wy,
			const real* ws,
			const real theta,
			const real* z,
			const real* x,
			const real* g,
			real* r,
			const cudaStream_t& stream
			)
		{
			if(col > 4) {
				int nblocky = 512 / 8;
				kernel1<8><<<dim3(iDivUp(nfree, nblocky)), dim3(8, nblocky), 0, stream>>>
					(nfree, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
			} else if(col > 2) {
				int nblocky = 512 / 4;
				kernel1<4><<<dim3(iDivUp(nfree, nblocky)), dim3(4, nblocky), 0, stream>>>
					(nfree, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
			} else if(col > 1) {
				int nblocky = 512 / 2;
				kernel1<2><<<dim3(iDivUp(nfree, nblocky)), dim3(2, nblocky), 0, stream>>>
					(nfree, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
			} else if(col == 1){
				int nblocky = 512 / 1;
				kernel1<1><<<dim3(iDivUp(nfree, nblocky)), dim3(1, nblocky), 0, stream>>>
					(nfree, col, head, m, iPitch, wa, wy, ws, theta, z, x, g, r);
			}
		}


	};
};