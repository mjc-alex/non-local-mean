#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void initialData(float *x, int n) {
	for (int i = 0; i < n; ++i) {
		x[i] = 3;
	}
}
__global__ void vectorSum(float *d_a, float *d_b, int nElem) {
	int i = threadIdx.x;
  if (i < nElem) {
//		d_a[i] = i + 100;
//		d_b[i] = i + 10;
	d_b[i] += d_a[i];
	}	
	printf("threadIdx.x = %d\n", i);
}
int main(int argc, char *argv[])
{
	const int nElem = 10;
	dim3 block(100);
	dim3 grid((nElem + block.x - 1) / block.x);
	float *d_a, *d_b, *h_a, *h_b, *h_c;
	int nBytes = nElem * sizeof(float);	

	printf("gpu: block.x = %d, grid.x = %d\n", block.x, grid.x);

	h_a = (float*)malloc(nBytes);
	h_b = (float*)malloc(nBytes);
	h_c = (float*)malloc(nBytes);

	initialData(h_a, nElem);
	initialData(h_b, nElem);

	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_b, nBytes);
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

//	for (int i = 0; i < nElem; ++i) {
//		if (i % 6 == 0) printf("\n");
//		printf("%d ", h_a[i]);
//	}
	vectorSum<<<grid, block>>>(d_a, d_b, nElem);	
	cudaMemcpy(h_c, d_b, nBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < nElem; ++i) {
		if (i % 6 == 0) printf("\n");
		printf("%f ", h_c[i]);
	}
	free(h_a);
	free(h_b);
	free(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaDeviceReset();

	return 0;
}

