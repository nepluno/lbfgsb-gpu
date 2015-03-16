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
	namespace matupd {

		__global__
		void kernel0
		(
		int n,
		real* wy,
		const real* r,
		const int iPitch)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;

			wy[i * iPitch] = r[i];
		}

		__global__
		void kernel1
		(
			real* sy,
			const int iPitch_i,
			const int iPitch_j,
			const int col
		)
		{
			const int i = threadIdx.x;
			const int j = threadIdx.y;

			if(j >= col - 1 || i >= j)
				return;

			sy[j * iPitch_i + i * iPitch_j] = sy[(j + 1) * iPitch_i + (i + 1) * iPitch_j];
		}

		template<int bx>
		__global__
		void kernel20(
			const int n,
			const int head,
			const int m,
			const int col,
			const int iPitch,
			const int oPitch,
			const real* d,
			real* buf_array_p,
			const real* wy)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			int pointr = Modular((head + j), m);
			if(i < n) {
				mySum = d[i] * wy[i * iPitch + pointr];
			} else {
				mySum = 0;
			}

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if (tid == 0) 
				buf_array_p[j * oPitch + blockIdx.x] = mySum;
		}

		template<int bx>
		__global__
		void kernel21(
			const int n,
			const int iPitch,
			const int oPitch,
			const real* buf_in,
			real* buf_out)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(i < n)
				mySum = buf_in[j * iPitch + i];
			else
				mySum = 0;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if(tid == 0) {
				buf_out[j * oPitch + blockIdx.x] = mySum;
			}
		}

		void prog0(
			const int& n,
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
			const int& iPitch_j,
			real* buf_array_p,
			cudaStream_t st)
		{
			CheckBuffer(wy, m, n * m);
			kernel0<<<dim3(iDivUp(n, 512)), dim3(512), 0, st>>>
				(n, wy + itail, r, iPitch0);
			CheckBuffer(wy, m, n * m);
/*
			if( iupdat > m )
			{
				kernel1<<<1, dim3(col - 1, col - 1), 0, st>>>
					(sy, iPitch_i, iPitch_j, col);
			}*/
			
			if(col > 1) {
				CheckBuffer(sy, iPitch_i, col * iPitch_i);
				int nblock0 = n;
				int mi = log2Up(nblock0);
				int nblock1 = iDivUp2(nblock0, mi);
			
				real* oFinal = sy + (col - 1) * iPitch_i;

				real* output = (nblock1 == 1) ? oFinal : buf_array_p;

				int op20 = (nblock1 == 1) ? iPitch_j : n;
				
				dynamicCall(kernel20, mi, nblock1, col - 1, st, (nblock0, head, m, col, iPitch0, op20, d, output, wy));

/*
				kernel20<<<dim3(nblock1, col - 1), dim3(512), 0, st>>>
					(nblock0, head, m, col, iPitch0, op20, d, output, wy);*/

				nblock0 = nblock1;
				//Launch Ker 0
				while(nblock0 > 1) {

					nblock1 = iDivUp2(nblock0, mi);

					real* input = output;

					output = (nblock1 == 1) ? oFinal : (output + nblock0);

					int op20 = (nblock1 == 1) ? iPitch_j : n;
					dynamicCall(kernel21, mi, nblock1, col - 1, st, (nblock0, n, op20, input, output));

/*
					kernel21<<<dim3(nblock1, col - 1), dim3(512), 0, st>>>
						(nblock0, n, op20, input, output);*/

					nblock0 = nblock1;
				}
				CheckBuffer(sy, iPitch_i, col * iPitch_i);
			}
			cudaMemcpyAsync(sy + (col - 1) * iPitch0 + col - 1, &dr, sizeof(real), cudaMemcpyHostToDevice, st);
			CheckBuffer(sy, iPitch_i, col * iPitch_i);
		}


	};
};