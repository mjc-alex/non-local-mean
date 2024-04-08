#ifndef _IMAGE_H_
#define _IMAGE_H_
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

typedef unsigned char uchar;

#define CUCALL(call) do{ cudaError_t err = call ; if (cudaSuccess != err){ fprintf(stderr, "cuda error at %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); fflush(stderr); exit(1); }}while(0)
#define CUCALL_RETURN(call,status) do{ cudaError_t err = call ; if (cudaSuccess != err){ fprintf(stderr, "cuda error at %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); fflush(stderr); status = 0; } else status = 1;}while(0)

struct Image {
	unsigned nvoxels;
	unsigned dims[3];
	float* data;
    size_t pitch;
	float* ptr;
};

void InitImage(struct Image* I, int h, int w, uchar buf[]);
double timestamp();
void destroyImage(struct Image* I);
void alloc_gpu(unsigned padding, struct Image *I);
void to_gpu(unsigned padding, struct Image *I);
void from_gpu(unsigned padding, struct Image *I);
#endif