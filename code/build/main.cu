#include<iostream>
#include<vector>
#include<cuda_runtime.h>
#include<cuda_profiler_api.h>
#include<cmath>
#include<sys/time.h>
#include"cuda_utility.h"
#include"main.h"

//#define _SHARED_
#define _GLOBAL_
#define	PARAMETER 10

#define BLOCK_X 32
#define BLOCK_Y 1

#define GRID_Y 64

#define SH_SIZE 8  //2 4 6 8 12 24

// #define CUCALL(call) do{\
// 	cudaError_t err = call ;\
// 	if (cudaSuccess != err){\
// 		fprintf(stderr, "cuda error at %s:%d, %s\n",\
// 				__FILE__, __LINE__, cudaGetErrorString(err));\
// 		fflush(stderr);\
// 		exit(1);\
// 	}\
// }while(0)
#ifndef _GLOBAL_
__global__ void ShMem(unsigned char *data,float *out_image,int W, int H, int sr_size, int nb_size,int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{
	// const unsigned ix = threadIdx.x + blockIdx.x * blockDim.x;
	// const unsigned iy = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned lx = blockIdx.x * blockDim.x;
	const unsigned ly = blockIdx.y * blockDim.y;
	const unsigned sr = (sr_size - 1) / 2;
	const unsigned nb = (nb_size - 1) / 2;

	const unsigned share_size = SH_SIZE + sr_size + nb_size - 2;
	const unsigned itra = share_size / SH_SIZE;
	extern __shared__ float smem[];
	//__shared__ float smem[8*1024];

	//copy global mem to shared mem
	for (int i = 4 * threadIdx.y; i < 4 * threadIdx.y + itra; i++) {
		for (int j = 4 * threadIdx.x; j < 4 * threadIdx.x + itra; j++) {
			smem[j + i * share_size] = data[lx + j + (i + ly) * W];
		}
	}

	__syncthreads();

	float sum = 0.0f;
	float weights = 0.0f;
	float weight = 0.0f;

	for (int i = 0; i < sr_size; i++) {
		for (int j = 0; j < sr_size; j++) {
			for(int k = 0; k < nb_size; k++) {
				for(int t = 0;t < nb_size; t++) {
					float d = smem[j + threadIdx.x + t + (i + threadIdx.y + k) * share_size] - smem[sr + threadIdx.x + t + (sr + threadIdx.y + k) * share_size];
					weight += d * d;
				}
			}
			weight = weight / (float) (nb_size * nb_size);
			weight = exp(-(weight * 0.01));
			weights += weight;
			sum += weight * smem[threadIdx.x + nb + j + (i + threadIdx.y + nb) * share_size];
			weight = 0;
		}
	}
	out_image[lx + threadIdx.x + (ly + threadIdx.y) * IMAGE_SIZE_X] = sum / weights;
//	out_image[lx + threadIdx.x + (ly + threadIdx.y) * IMAGE_SIZE_X] = smem[threadIdx.x + sr + nb + (threadIdx.y + sr + nb) * share_size]; 
}
#else
__global__ void NLmeansOnGPU(unsigned char *data,float *out_image,int W, int H, int sr_size, int nb_size,int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{	
	unsigned int col = IMAGE_SIZE_Y/GRID_Y;
	unsigned int ix = threadIdx.x +blockIdx.x*32;	
	unsigned int iy = blockIdx.y;	
	unsigned int size = (sr_size - 1) / 2 + (nb_size - 1) / 2;

	float g1 = 0;
	float g3 = 0;
	float gmid = 0;
	float result = 0;
//out_image
	for(int itra = 0; itra < col; itra++){
		for (int i = 0; i < sr_size; i++)
		{
			for (int j = 0; j < sr_size; j++)
			{
				for(int k = 0; k < nb_size; k++)
				{
					for(int t = 0;t < nb_size; t++)
					{
						g1 = data[size + ix +t+ ( k + size + itra+iy*col) * W] - data[ix + j +t+ ( k + i + itra+iy*col) * W];
						g1 = g1 * g1;
						g3+=g1;
					}
				} 
				g3 = g3 / (float)(nb_size * nb_size);
				g3 = exp(-g3 * 0.01);
				gmid += g3;
				result += g3 * data[ix + j + (i + itra+iy*col)* W];
				g3 = 0;
		   	}
		}
	//	printf("%f,%f\n",gmid,result);
		out_image[ix+ (itra+iy*col)*IMAGE_SIZE_X] = result / gmid;
		result = 0;
		gmid = 0;
	}
}
#endif
void NLMeansProcessor::NL_Means(unsigned char *GPU_input, float *GPU_result, int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{
#ifdef _GLOBAL_
	unsigned char *data = nullptr;
	float *out_image = nullptr;

	cudaMallocCheck((void**)&data, W * H * sizeof(unsigned char));
	cudaMallocCheck((void**)&out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float));

	cudaMemcpyCheck(data, GPU_input, W * H * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int GRID_X=IMAGE_SIZE_X/BLOCK_X;
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(GRID_X, GRID_Y);

	for (int i = 0; i < 100; i++)
		NLmeansOnGPU <<<grid, block >>> (data, out_image, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	

	cudaMemcpyCheck(GPU_result, out_image,IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFreeCheck(data);
	cudaFreeCheck(out_image);

	cudaDeviceSynchronize();
	cudaProfilerStop();
#else
	unsigned char *data = nullptr;
	float *out_image = nullptr;

	cudaMallocCheck((void**)&data, W * H * sizeof(unsigned char));
	cudaMallocCheck((void**)&out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float));

	cudaMemcpyCheck(data, GPU_input, W * H * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 block(SH_SIZE, SH_SIZE);
	dim3 grid(IMAGE_SIZE_X / SH_SIZE, IMAGE_SIZE_Y / SH_SIZE);

	//cudaFuncSetAttribute(ShMem, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * (1 << 10));

	//ShMem <<<grid, block, 32 * 32 * sizeof(float)>>> (data, out_image, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	
	const unsigned share_size = SH_SIZE + sr_size + nb_size - 2;

	for (int i = 0; i < 100; i++)
		//ShMem <<<grid, block, 32 * 32 * sizeof(float)>>> (data, out_image, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
		ShMem <<<grid, block, share_size * share_size * sizeof(float)>>> (data, out_image, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
		//ShMem <<<grid, block>>> (data, out_image, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	cudaMemcpyCheck(GPU_result, out_image,IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFreeCheck(data);
	cudaFreeCheck(out_image);

	cudaDeviceSynchronize();
	cudaProfilerStop();
#endif

}
