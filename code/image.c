#include "image.h"

void InitImage(struct Image* I, int h, int w, uchar buf[]) {
    I -> dims[0] = h;
    I -> dims[1] = w;
    I -> dims[2] = 1;
    I -> nvoxels = h * w;
    I -> data = (float*)malloc(h*w*sizeof(float));
    for (unsigned i=0; i<I->nvoxels; i++) I->data[i] = (float)buf[i];
}
double timestamp()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + tv.tv_usec * 1e-6;
}
void destroyImage(struct Image* I) {
    free(I -> data);
}

cudaPitchedPtr alloc3d(size_t size, size_t X, size_t Y, size_t Z)
{
	cudaPitchedPtr ptr;
	cudaExtent vol = make_cudaExtent(size*X, Y, Z);
	CUCALL(cudaMalloc3D(&ptr, vol));
	CUCALL(cudaMemset(ptr.ptr, 0x00, ptr.pitch*Y*Z));
	return ptr;
}

void free3d(cudaPitchedPtr* ptr)
{
	CUCALL(cudaFree(ptr->ptr));
}

void memcpy3d_to(
	void* cpu_ptr, cudaPitchedPtr gpu_ptr, size_t size,
	unsigned X, unsigned Y, unsigned Z, unsigned R)
{
	cudaMemcpy3DParms params = {0};
	params.srcPtr = make_cudaPitchedPtr(cpu_ptr, size*X, X, Y);
	params.srcPos = make_cudaPos(0, 0, 0);
	params.dstPtr = gpu_ptr;
	params.dstPos = make_cudaPos(R*sizeof(float), R, R);
	params.extent = make_cudaExtent(size*X, Y, Z);
	params.kind = cudaMemcpyHostToDevice;
	CUCALL(cudaMemcpy3D(&params));
}

void memcpy3d_from(
	cudaPitchedPtr gpu_ptr, void* cpu_ptr, size_t size,
	unsigned X, unsigned Y, unsigned Z, unsigned R)
{
	cudaMemcpy3DParms params = {0};
	params.srcPtr = gpu_ptr;
	params.srcPos = make_cudaPos(R*sizeof(float), R, R);
	params.dstPtr = make_cudaPitchedPtr(cpu_ptr, size*X, X, Y);
	params.dstPos = make_cudaPos(0, 0, 0);
	params.extent = make_cudaExtent(size*X, Y, Z);
	params.kind = cudaMemcpyDeviceToHost;
	CUCALL(cudaMemcpy3D(&params));
}
void alloc_gpu(unsigned padding, struct Image *I)
{
	const unsigned P = 2*padding;
	if (ptr != NULL) free3d(&cuda_ptr);
	cuda_ptr = alloc3d(sizeof(float), P+I->dims[0], P+I->dims[1], P+I->dims[2]);
	ptr = (float*)cuda_ptr.ptr;
	pitch = cuda_ptr.pitch / sizeof(float);
}
void to_gpu(unsigned padding, struct Image *I)
{
	memcpy3d_to(
		(void*)I->data, cuda_ptr, sizeof(float),
		I->dims[0], I->dims[1], I->dims[2], padding);
}

void from_gpu(unsigned padding, struct Image *I)
{
	memcpy3d_from(
		cuda_ptr, (void*)I->data, sizeof(float),
		I->dims[0], I->dims[1], I->dims[2], padding);
}