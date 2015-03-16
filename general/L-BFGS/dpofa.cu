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
	namespace dpofa {

		#define CUDA_BLOCK_SIZE 16
		
		__global__ void cuda_chol_iter(real* m, int n, int boffset) {
			int k;
			int x = threadIdx.x ;
			int y = threadIdx.y ;
			int bsize = blockDim.x ;
			__shared__ real b[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE+1] ;
			b [x][y] = m[ ( x + boffset ) * n + boffset + y ];
			for ( k = 0; k < bsize; k++) {
				__syncthreads();
				if( x == k ) {
					if(b[x][x] < machineepsilon)
						b[x][x] = machineepsilon;
					real fac = sqrtr(b[x][x]);
					if ( y >= x ) {
						b[x][y] /= fac;
					}
				}
				__syncthreads();
				if ( x > k && y >= x ) {
					b [x][y] -= b[k][y] * b[k][x];
				}
			}
			__syncthreads();
			m[ (boffset + x) * n + boffset + y ] = b[x][y];
		}

		void prog0(real* m, int n, int pitch, int boffset, const cudaStream_t& st) {
			cuda_chol_iter<<<1, dim3(n, n), 0, st>>>
				(m, pitch, boffset);
		}
	}
}