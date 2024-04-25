// cuda_utility.h

#pragma once

#include <iostream>
#include <cuda_runtime.h>

// 定义 CUDA 错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t cudaStatus = call; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
        return cudaStatus; \
    } \
} while (0)

// 封装 CUDA 内存分配函数，包含错误检查
// 函数返回 cudaError_t 类型，表示 CUDA 函数执行的状态
// 参数 devPtr 是设备内存指针，size 是要分配的内存大小
cudaError_t cudaMallocCheck(void** devPtr, size_t size);

// 封装 CUDA 内存拷贝函数，包含错误检查
// 函数返回 cudaError_t 类型，表示 CUDA 函数执行的状态
// 参数 dst 是目标地址，src 是源地址，count 是要拷贝的字节数，kind 是拷贝的类型
cudaError_t cudaMemcpyCheck(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

// 封装 CUDA 内存释放函数，包含错误检查
// 函数返回 cudaError_t 类型，表示 CUDA 函数执行的状态
// 参数 devPtr 是要释放的设备内存指针
cudaError_t cudaFreeCheck(void* devPtr);
