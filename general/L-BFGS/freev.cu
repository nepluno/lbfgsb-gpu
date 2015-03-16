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
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace lbfgsbcuda {
	__global__
	void kernel0(
	int* index,
	const int* iwhere,
	int* temp_ind1,
	int* temp_ind2,
	int nfree,
	int n
	)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;

		if(i >= n)
			return;

		int k = index[i];
		int iwk = iwhere[k];
		int t1, t2;

		if(i < nfree && iwk > 0) {
			t1 = 1;
			t2 = 0;
		} else if(i >= nfree && iwk <= 0) {
			t1 = 0;
			t2 = 1;
		} else {
			t1 = t2 = 0;
		}

		temp_ind1[i] = t1;
		temp_ind2[i] = t2;
	}

	__global__
	void kernel1(
	const int* index,
	const int* temp_ind1,
	const int* temp_ind2,
	const int* temp_ind3,
	const int* temp_ind4,
	int* indx2,
	int n
	) 
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;

		if(i >= n)
			return;

		int k = index[i];
		if(temp_ind1[i]) 
		{
			indx2[n - temp_ind3[i]] = k;
		} else if(temp_ind2[i]) {
			indx2[temp_ind4[i] - 1] = k;
		}
	}

	__global__
	void kernel2(
	const int* iwhere,
	int* temp_ind1,
	int* temp_ind2,
	int n
	) 
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;

		if(i >= n)
			return;

		int iwi = iwhere[i];
		if(iwi <= 0) 
		{
			temp_ind1[i] = 1;
			temp_ind2[i] = 0;
		} else {
			temp_ind1[i] = 0;
			temp_ind2[i] = 1;
		}
	}

	__global__
	void kernel3(
	int* index,
	const int* iwhere,
	const int* temp_ind1,
	const int* temp_ind2,
	int n
	) 
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;

		if(i >= n)
			return;

		int iwi = iwhere[i];
		if(iwi <= 0) 
		{
			index[temp_ind1[i] - 1] = i;
		} else {
			index[n - temp_ind2[i]] = i;
		}
	}

	namespace freev {
		void prog0( 
			const int& n, 
			int& nfree, 
			int* index, 
			int& nenter, 
			int& ileave, 
			int* indx2, 
			const int* iwhere, 
			bool& wrk, 
			const bool& updatd, 
			const bool& cnstnd, 
			const int& iter,
			int* temp_ind1,
			int* temp_ind2,
			int* temp_ind3,
			int* temp_ind4
			)
		{
			nenter = -1;
			ileave = n;
			if( iter > 0 && cnstnd )
			{
				CheckBuffer_int(iwhere, n, n);
				CheckBuffer_int(index, n, n);

				kernel0<<<iDivUp(n, 512), 512>>>
					(index, iwhere, temp_ind1, temp_ind2, nfree, n);

				CheckBuffer_int(temp_ind1, n, n);
				CheckBuffer_int(temp_ind2, n, n);

				thrust::device_ptr<int> dptr_ind1(temp_ind1);
				thrust::device_ptr<int> dptr_ind2(temp_ind2);
				thrust::device_ptr<int> dptr_ind3(temp_ind3);
				thrust::device_ptr<int> dptr_ind4(temp_ind4);

				thrust::inclusive_scan(dptr_ind1, dptr_ind1 + n, dptr_ind3);
				thrust::inclusive_scan(dptr_ind2, dptr_ind2 + n, dptr_ind4);

				CheckBuffer_int(temp_ind3, n, n);
				CheckBuffer_int(temp_ind4, n, n);

				kernel1<<<iDivUp(n, 512), 512>>>
					(index, temp_ind1, temp_ind2, temp_ind3, temp_ind4, indx2, n);

				CheckBuffer_int(index, n, n);
				CheckBuffer_int(indx2, n, n);

				cutilSafeCall(cudaMemcpy(&ileave, temp_ind3 + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaMemcpy(&nenter, temp_ind4 + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
				ileave = n - ileave;
				nenter = nenter - 1;
			}

			wrk = ileave < n || nenter >= 0 || updatd;

			CheckBuffer_int(iwhere, n, n);

			kernel2<<<iDivUp(n, 512), 512>>>
				(iwhere, temp_ind1, temp_ind2, n);

			CheckBuffer_int(temp_ind1, n, n);
			CheckBuffer_int(temp_ind2, n, n);

			thrust::device_ptr<int> dptr_ind1(temp_ind1);
			thrust::device_ptr<int> dptr_ind2(temp_ind2);

			thrust::inclusive_scan(dptr_ind1, dptr_ind1 + n, dptr_ind1);
			thrust::inclusive_scan(dptr_ind2, dptr_ind2 + n, dptr_ind2);

			CheckBuffer_int(temp_ind1, n, n);
			CheckBuffer_int(temp_ind2, n, n);

			kernel3<<<iDivUp(n, 512), 512>>>
				(index, iwhere, temp_ind1, temp_ind2, n);

			CheckBuffer_int(index, n, n);

			cutilSafeCall(cudaMemcpy(&nfree, temp_ind1 + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
		}
	};
};
