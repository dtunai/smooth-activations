#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "smelu.h"

// construct CUDA kernel template for the forward pass
template <typename T>
__global__ void smelu_forward_kernel(const T* input, T* output, const T* alpha, const size_t size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const T x = input[i];
        output[i] = x >= 0 ? x : alpha[i] * (exp(x) - 1);
    }
}

// construct CUDA kernel template for the backward pass
template <typename T>
__global__ void smelu_backward_kernel(const T* input, const T* grad_output, T* grad_input, const T* alpha, const size_t size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const T x = input[i];
        grad_input[i] = x >= 0 ? grad_output[i] : grad_output[i] * alpha[i] * exp(x);
    }
}

// construct CUDA kernel template for SmeLU forward pass function
template <typename T>
std::vector<T> SmeLU<T>::forward(const std::vector<T>& input) {
    const size_t size = input.size();
    this->alpha.resize(size);

    T *d_input, *d_output, *d_alpha;
    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, size * sizeof(T));
    cudaMalloc(&d_alpha, size * sizeof(T));

    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, this->alpha.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int num_blocks = static_cast<int>((size + block_size - 1) / block_size);
    smelu_forward_kernel<T><<<num_blocks, block_size>>>(d_input, d_output, d_alpha, size);

    std::vector<T> output(size);
    cudaMemcpy(output.data(), d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_alpha);

    return output;
}

// construct SmeLU backward pass function
template <typename T>
std::vector<T> SmeLU<T>::backward(const std::vector<T>& input, const std::vector<T>& grad_output) {
    const size_t size = input.size();
    this->alpha.resize(size);

    T *d_input, *d_grad_output, *d_grad_input, *d_alpha;
    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_grad_output, size * sizeof(T));
    cudaMalloc(&d_grad_input, size * sizeof(T));
    cudaMalloc(&d_alpha, size * sizeof(T));

    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, this->alpha.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int num_blocks = static_cast<int>((size + block_size - 1) / block_size);
    smelu_backward_kernel<<<num_blocks, block_size>>>(d_input, d_grad_output, d_grad_input, d_alpha, size);

    std::vector<T> grad_input(size);
    cudaMemcpy(grad_input.data(), d_grad_input, size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_alpha);

    return grad_input;
}

// init SmeLU templates
template class SmeLU<float>;
template class SmeLU<double>;
