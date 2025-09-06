#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>

// Helper macro for checking CUDA API calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * @brief CUDA kernel for element-wise vector addition.
 *
 * @param x Pointer to the first input vector on the device.
 * @param y Pointer to the second input vector on the device.
 * @param output Pointer to the output vector on the device.
 * @param n_elements The total number of elements in the vectors.
 */
__global__ void addKernel(const float* x, const float* y, float* output, int n_elements) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure the thread is within the bounds of the array
    if (idx < n_elements) {
        output[idx] = x[idx] + y[idx];
    }
}


int main() {
    // ====================================================================
    // 1. Verification Step (like the first part of the Python script)
    // ====================================================================
    std::cout << "--- Running Verification ---" << std::endl;
    const int verify_size = 98432;

    // --- Host (CPU) Memory Allocation and Initialization ---
    std::vector<float> h_x(verify_size);
    std::vector<float> h_y(verify_size);
    std::vector<float> h_output_gpu(verify_size);
    std::vector<float> h_output_cpu(verify_size);

    // Initialize input vectors with random data (similar to torch.rand)
    std::default_random_engine generator(0); // Seed with 0
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < verify_size; ++i) {
        h_x[i] = distribution(generator);
        h_y[i] = distribution(generator);
    }

    // --- Device (GPU) Memory Allocation ---
    float *d_x, *d_y, *d_output;
    gpuErrchk(cudaMalloc((void**)&d_x, verify_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_y, verify_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_output, verify_size * sizeof(float)));

    // --- Copy Data from Host to Device ---
    gpuErrchk(cudaMemcpy(d_x, h_x.data(), verify_size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y.data(), verify_size * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    int blockSize = 1024; // Number of threads per block (same as BLOCK_SIZE in Triton)
    // Number of blocks needed = ceil(n_elements / blockSize)
    int numBlocks = (verify_size + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_x, d_y, d_output, verify_size);

    // --- Copy Result from Device to Host ---
    gpuErrchk(cudaMemcpy(h_output_gpu.data(), d_output, verify_size * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Verification ---
    float max_diff = 0.0f;
    for (int i = 0; i < verify_size; ++i) {
        h_output_cpu[i] = h_x[i] + h_y[i];
        float diff = std::abs(h_output_cpu[i] - h_output_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "First 5 elements (CPU):   ";
    for(int i=0; i<5; ++i) std::cout << h_output_cpu[i] << " ";
    std::cout << std::endl;

    std::cout << "First 5 elements (GPU):   ";
    for(int i=0; i<5; ++i) std::cout << h_output_gpu[i] << " ";
    std::cout << std::endl;

    std::cout << "Max difference = " << max_diff << std::endl;
    std::cout << std::endl;

    // --- Cleanup Device Memory ---
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_y));
    gpuErrchk(cudaFree(d_output));


    // ====================================================================
    // 2. Benchmarking Step
    // ====================================================================
    std::cout << "--- Running Benchmark ---" << std::endl;
    printf("%-20s %-15s %-15s\n", "Size", "CUDA (GB/s)", "CPU (GB/s)");
    printf("--------------------------------------------------\n");

    for (int p = 12; p < 28; ++p) {
        long long n_elements = 1LL << p;

        // --- Allocate host memory ---
        std::vector<float> host_x(n_elements);
        std::vector<float> host_y(n_elements);
        // No need for output on host for benchmark

        // --- Allocate device memory ---
        float *dev_x, *dev_y, *dev_out;
        gpuErrchk(cudaMalloc((void**)&dev_x, n_elements * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&dev_y, n_elements * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&dev_out, n_elements * sizeof(float)));

        // --- Copy to device ---
        gpuErrchk(cudaMemcpy(dev_x, host_x.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_y, host_y.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));

        // --- CUDA Benchmark ---
        int grid_size = (n_elements + blockSize - 1) / blockSize;
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        // Warm-up run
        addKernel<<<grid_size, blockSize>>>(dev_x, dev_y, dev_out, n_elements);

        gpuErrchk(cudaEventRecord(start));
        // Perform multiple iterations for a more stable measurement if needed
        addKernel<<<grid_size, blockSize>>>(dev_x, dev_y, dev_out, n_elements);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));

        float milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

        // Bandwidth = (bytes_read + bytes_written) / time_in_seconds
        // 3 tensors (x, y, output) * num_elements * sizeof(float)
        double gpu_gbps = (3.0 * n_elements * sizeof(float)) / (milliseconds / 1000.0) / 1e9;
        
        // --- CPU Benchmark (for comparison, like 'torch' provider) ---
        auto cpu_start = std::chrono::high_resolution_clock::now();
        std::vector<float> cpu_out(n_elements);
        for(long long i = 0; i < n_elements; ++i) {
            cpu_out[i] = host_x[i] + host_y[i];
        }
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_ms = cpu_stop - cpu_start;
        double cpu_gbps = (3.0 * n_elements * sizeof(float)) / (cpu_ms.count() / 1000.0) / 1e9;


        printf("%-20lld %-15.2f %-15.2f\n", n_elements, gpu_gbps, cpu_gbps);
        
        // --- Cleanup for this iteration ---
        gpuErrchk(cudaFree(dev_x));
        gpuErrchk(cudaFree(dev_y));
        gpuErrchk(cudaFree(dev_out));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
    }

    return 0;
}
