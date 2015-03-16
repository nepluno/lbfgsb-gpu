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
	namespace subsm {
		
		template<int bx>
		__global__
		void kernel00(
			const int nsub,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			const int oPitch,
			real* buf_array_p,
			const real* wy,
			const real* ws,
			const real* d,
			const real theta
			)
		{
			const int j = blockIdx.x * blockDim.x + threadIdx.x;
			const int i = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(j < nsub) {
				int pointr = Modular((head + i % col), m);

				if(i >= col) {
					mySum = ws[j * iPitch_ws + pointr] * theta;
				} else {
					mySum = wy[j * iPitch_ws + pointr];
				}
				mySum *= d[j];
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
				buf_array_p[i * oPitch + blockIdx.x] = mySum;
		}

		template<int bx>
		__global__
		void kernel01(
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
			const int n,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			real* buf_array_p,
			const real* wy,
			const real* ws,
			const real* d,
			real* wv,
			const real theta,
			const cudaStream_t& stream
			)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output = (nblock1 == 1) ? wv : buf_array_p;
			int op20 = (nblock1 == 1) ? 1 : n;

			dynamicCall(kernel00, mi, nblock1, col * 2, stream, (n, head, m, col, iPitch_ws, op20, output, wy, ws, d, theta));

			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input = output;

				output = (nblock1 == 1) ? wv : (output + nblock0);

				int op20 = (nblock1 == 1) ? 1 : n;
				dynamicCall(kernel01, mi, nblock1, col * 2, stream, (nblock0, n, op20, input, output));

				nblock0 = nblock1;
			}
		}

		__global__
		void kernel1(
			real* wv) 
		{
			const int i = threadIdx.x;
			wv[i] = -wv[i];
		}

		void prog1(
			real* wn,
			int col,
			int iPitch_wn,
			real* wv,
			const cudaStream_t& stream
			)
		{
			int col2 = col * 2;
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);

			cublasSetStream(cublasHd, stream);
			cublasRtrsv(
				cublasHd, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 
				CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
			lbfgsbcuda::CheckBuffer(wn, iPitch_wn, iPitch_wn * 7);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			kernel1<<<1, col, 0, stream>>>
				(wv);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			cublasRtrsv(cublasHd, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, 
				CUBLAS_DIAG_NON_UNIT, col2, wn, iPitch_wn, wv, 1);
			lbfgsbcuda::CheckBuffer(wv, col * 2, col * 2);
			cublasSetStream(cublasHd, NULL);
		}

		template<int bsize>
		__global__
		void kernel2(
		int nsub,
		const int col,
		const int head,
		const int m,
		const int iPitch,
		const real* wv,
		const real* wy,
		const real* ws,
		const real inv_theta,
		real* d
		)
		{
			const int i = blockIdx.x * blockDim.y + threadIdx.y;
			const int tidx = threadIdx.x; //8
			const int tidy = threadIdx.y; //64
			
			volatile __shared__ real sdata[(512 / bsize)][bsize + 1];

			__shared__ real a[2][bsize+1];

			real mySum;

			if(tidy == 0 && tidx < col) {
				a[0][tidx] = wv[tidx] * inv_theta;
				a[1][tidx] = wv[col + tidx];
			}

			if(i < nsub && tidx < col) {
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

			if(tidx == 0 && i < nsub) {
				d[i] = (d[i] + mySum) * inv_theta;
			}
		}

		void prog2(
			const int nsub,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const real* wv,
			const real* wy,
			const real* ws,
			const real theta,
			real* d,
			const cudaStream_t& stream
			)
		{
			real invtheta = 1.0 / theta;

			if(col > 4) {
				int nblocky = 512 / 8;
				kernel2<8><<<dim3(iDivUp(nsub, nblocky)), dim3(8, nblocky), 0, stream>>>
					(nsub, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col > 2) {
				int nblocky = 512 / 4;
				kernel2<4><<<dim3(iDivUp(nsub, nblocky)), dim3(4, nblocky), 0, stream>>>
					(nsub, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col > 1) {
				int nblocky = 512 / 2;
				kernel2<2><<<dim3(iDivUp(nsub, nblocky)), dim3(2, nblocky), 0, stream>>>
					(nsub, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			} else if(col == 1){
				int nblocky = 512 / 1;
				kernel2<1><<<dim3(iDivUp(nsub, nblocky)), dim3(1, nblocky), 0, stream>>>
					(nsub, col, head, m, iPitch, wv, wy, ws, invtheta, d);
			}
		}

		__device__
		inline void minex(volatile real& a, volatile real& b, volatile int& ia, volatile int& ib)
		{
			if(a > b) {
				ia = ib, a = b;
			}
		}

		template<int bx>
		__global__
		void kernel30(
		const int nsub,
		real* d,
		const int* nbd,
		real* t,
		int* ti,
		real* x,
		const real* u,
		const real* l
		)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			const int tid = threadIdx.x;

			volatile __shared__ real sdata[bx];
			volatile __shared__ int sdatai[bx];

			real mySum = 1.0;

			if(i < nsub) {
				const int nbdi = nbd[i];

				if(nbdi != 0) {
					real dk = d[i];
				    if( dk < 0 && nbdi <= 2 )
					{
						real temp2 = l[i] - x[i];
						if( temp2 >= 0 )
						{
							mySum = 0;
						}
						else
						{
							mySum = minr(1.0, temp2 / dk);
						}
					}
					else if( dk > 0 && nbdi >= 2 )
					{
						real temp2 = u[i] - x[i];
						if( temp2 <= 0 )
						{
							mySum = 0;
						}
						else
						{
							mySum = minr(1.0, temp2 / dk);
						}
					}
				}
			}


			sdata[tid] = mySum;
			sdatai[tid] = i;
			__syncthreads();

			t[i] = mySum;
			ti[i] = i;

			if(bx > 512) {if (tid < 512) { minex(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { minex(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { minex(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { minex(sdata[tid], sdata[tid +  64], sdatai[tid], sdatai[tid + 64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				volatile int* smemi = sdatai + tid;
				if(bx > 32) {minex(*smem, smem[32], *smemi, smemi[32]);}
				if(bx > 16) {minex(*smem, smem[16], *smemi, smemi[16]);}
				if(bx > 8) {minex(*smem, smem[8], *smemi, smemi[8]);}
				if(bx > 4) {minex(*smem, smem[4], *smemi, smemi[4]);}
				if(bx > 2) {minex(*smem, smem[2], *smemi, smemi[2]);}
				if(bx > 1) {minex(*smem, smem[1], *smemi, smemi[1]);}
								
				if (tid == 0) {
					t[blockIdx.x] = *smem;
					ti[blockIdx.x] = *smemi;

					if(gridDim.x == 1 && *smem < 1) {
						real dk = d[*smemi];
						if(dk > 0) {
							x[*smemi] = u[*smemi];
							d[*smemi] = 0;
						} else if(dk < 0)
						{
							x[*smemi] = l[*smemi];
							d[*smemi] = 0;
						}
					}
				}
			}
		}

		template<int bx>
		__global__
		void kernel31(
			const int n,
			const real* buf_in,
			const int* bufi_in,
			real* buf_out,
			int* bufi_out,
			real* d,
			real* x,
			const real* u,
			const real* l
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];
			volatile __shared__ int sdatai[bx];

			real mySum;
			int mySumi;
			if(i < n) {
				mySum = buf_in[i];
				mySumi = bufi_in[i];
			} else {
				mySum = 1.0;
				mySumi = 0;
			}

			sdata[tid] = mySum;
			sdatai[tid] = mySumi;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { minex(sdata[tid], sdata[tid + 512], sdatai[tid], sdatai[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { minex(sdata[tid], sdata[tid + 256], sdatai[tid], sdatai[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { minex(sdata[tid], sdata[tid + 128], sdatai[tid], sdatai[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { minex(sdata[tid], sdata[tid +  64], sdatai[tid], sdatai[tid + 64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				volatile int* smemi = sdatai + tid;
				if(bx > 32) {minex(*smem, smem[32], *smemi, smemi[32]);}
				if(bx > 16) {minex(*smem, smem[16], *smemi, smemi[16]);}
				if(bx > 8) {minex(*smem, smem[8], *smemi, smemi[8]);}
				if(bx > 4) {minex(*smem, smem[4], *smemi, smemi[4]);}
				if(bx > 2) {minex(*smem, smem[2], *smemi, smemi[2]);}
				if(bx > 1) {minex(*smem, smem[1], *smemi, smemi[1]);}
								
				if (tid == 0) {
					buf_out[blockIdx.x] = *smem;
					bufi_out[blockIdx.x] = *smemi;
					
					if(gridDim.x == 1 && *smem < 1) {
						real dk = d[*smemi];
						if(dk > 0) {
							x[*smemi] = u[*smemi];
							d[*smemi] = 0;
						} else if(dk < 0)
						{
							x[*smemi] = l[*smemi];
							d[*smemi] = 0;
						}
					}
				}
			}
		}

		__global__
		void kernel32(
			const int nsub,
			real* x,
			const real* d,
			const real* alpha
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			__shared__ real salpha[1];
			
			if(i >= nsub)
				return;

			if(threadIdx.x == 0) {
				*salpha = alpha[0];
			}
			real xi = x[i];
			real di = d[i];

			__syncthreads();
			
			x[i] = salpha[0] * di + xi;
		}

		void prog3
		(
			const int nsub,
			real* d,
			const int* nbd,
			real* buf_s_r,
			int* bufi_s_r,
			real* x,
			const real* u,
			const real* l,
			const cudaStream_t& stream
		)
		{
			//kernel30(nsub, d, nbd, buf_s_r, bufi_s_r, x, u, l, alpha);
			int nblock0 = nsub;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output_r = buf_s_r;
			int* output_i = bufi_s_r;

			dynamicCall(kernel30, mi, nblock1, 1, stream, (nsub, d, nbd, output_r, output_i, x, u, l));

/*
			kernel30<<<dim3(nblock1), dim3(512)>>>
				(nsub, d, nbd, output_r, output_i, x, u, l);*/
			
			CheckBuffer_int(output_i, nsub, nsub);
			CheckBuffer(output_r, nsub, nsub);
			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input_r = output_r;
				int* input_i = output_i;

				output_r = output_r + nblock0;
				output_i = output_i + nblock0;

				dynamicCall(kernel31, mi, nblock1, 1, stream, (nblock0, input_r, input_i, output_r, output_i, d, x, u, l));

/*
				kernel31<<<dim3(nblock1), dim3(512)>>>
					(nblock0, input_r, input_i, output_r, output_i, d, x, u, l);*/

				nblock0 = nblock1;
			}

			kernel32<<<dim3(iDivUp(nsub, 512)), dim3(512), 0, stream>>>
				(nsub, x, d, output_r);

		}

	};
};