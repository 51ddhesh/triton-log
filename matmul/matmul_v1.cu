#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <algorithm>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32
#define WARP_SIZE 32


// Error checking macros
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)


#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)


// Device functions
__device__ __forceinline__ half leaky_relu(half x) {
    return x >= __float2half(0.0f) ? x : __hmul(x, __float2half(0.0f));
}

__device__ __forceinline__ float leaky_relu(float x) {
    return x >= 0.0f ? x : 0.01f * x;
}

// Optimized CUDA kernel for matrix multiplication
template <int ACTIVATION_TYPE> 
__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K, 
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn
) {
    // Shared memory for tiles
    __shared__ half As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ half Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // 

}

