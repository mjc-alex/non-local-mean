// cuda_utility.cpp

#include "cuda_utility.h"

// 实现 CUDA 内存分配函数，包含错误检查
cudaError_t cudaMallocCheck(void** devPtr, size_t size) {
	CUDA_CHECK(cudaMalloc(devPtr, size));
	return cudaSuccess;
}

// 实现 CUDA 内存拷贝函数，包含错误检查
cudaError_t cudaMemcpyCheck(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
	CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
	return cudaSuccess;
}


// 实现 CUDA 内存释放函数，包含错误检查
cudaError_t cudaFreeCheck(void* devPtr) {
	CUDA_CHECK(cudaFree(devPtr));
	return cudaSuccess;
}
