#include "../chubarov.h"

__global__ void EvalGradLMulKernel(std::size_t N,
                                   std::size_t M,
                                   std::size_t L, 
                                   float* grads,
                                   const float* sibling_values,
                                   const float* parent_grads) {

    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t l = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && l < L) {
        float grads_value = 0.0f;

        for (std::size_t m = 0; m < M; m++) {
            float parent_grad =  parent_grads[n * M + m];
            float local_grad = sibling_values[l * M + m];
            grads_value += local_grad * parent_grad;
        }
        grads[n * L + l] += grads_value;
    }
}

// FIXME: errorcheck!!
int Chubarov_EvalGradLMul(std::size_t N,
                          std::size_t M,
                          std::size_t L, 
                          float* grads,
                          const float* sibling_values,
                          const float* parent_grads) {

    float* d_sibling_values = nullptr;
    float* d_parent_grads   = nullptr;
    float* d_grads          = nullptr;

    std::size_t sibling_size = L * M * sizeof(float);
    std::size_t parent_size  = N * M * sizeof(float);
    std::size_t grads_size   = N * L * sizeof(float);

    cudaMalloc(&d_sibling_values, sibling_size);
    cudaMalloc(&d_parent_grads,    parent_size);
    cudaMalloc(&d_grads,            grads_size);

    cudaMemcpy(d_sibling_values, sibling_values, sibling_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent_grads,   parent_grads,    parent_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads,          grads,            grads_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (L + blockDim.y - 1) / blockDim.y);

    EvalGradLMulKernel<<<gridDim, blockDim>>>(N, M, L, d_grads, d_sibling_values, d_parent_grads);

    cudaMemcpy(grads, d_grads, grads_size, cudaMemcpyDeviceToHost);

    cudaFree(d_sibling_values);
    cudaFree(d_parent_grads);
    cudaFree(d_grads);

    return 0;
}


__global__ void EvalGradRMulKernel(float* sibling_values, float* parent_grads, float* grads, std::size_t N, std::size_t M, std::size_t L) {
    std::size_t l = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t m = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < L && m < M) {
        float grad_value = 0.0f;
        for (std::size_t n = 0; n < N; n++) {
            float local_grad = sibling_values[n * L + l];
            float parent_grad = parent_grads[n * M + m];
            grad_value += local_grad * parent_grad;
        }
        grads[l * M + m] = grad_value;
    }
}

int Chubarov_EvalGradRMul(std::size_t N,
                           std::size_t M,
                           std::size_t L, 
                           float* grads,
                           const float* sibling_values,
                           const float* parent_grads) {
    float* d_sibling_values = nullptr;
    float* d_parent_grads   = nullptr;
    float* d_grads          = nullptr;

    std::size_t sibling_size = N * L * sizeof(float);
    std::size_t parent_size  = N * M * sizeof(float);
    std::size_t grads_size   = L * M * sizeof(float);

    cudaMalloc(&d_sibling_values, sibling_size);
    cudaMalloc(&d_parent_grads,    parent_size);
    cudaMalloc(&d_grads,            grads_size);

    cudaMemcpy(d_sibling_values, sibling_values, sibling_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent_grads,     parent_grads,  parent_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads,                   grads,   grads_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); // Adjust according to your needs
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    EvalGradRMulKernel<<<gridDim, blockDim>>>(d_sibling_values, d_parent_grads, d_grads, N, M, L);

    cudaMemcpy(grads, d_grads, grads_size, cudaMemcpyDeviceToHost);

    cudaFree(d_sibling_values);
    cudaFree(d_parent_grads);
    cudaFree(d_grads);

    return 0;
}


__global__ void MulKernel(std::size_t N, 
                          std::size_t M,
                          std::size_t L, 
                          float* output,
                          const float* first,
                          const float* second) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (n < N && m < M) {
        float value = 0.0f;
        for (std::size_t l = 0; l < L; l++) {
            value += first[n * L + l] * second[l * M + m];
        }
        output[n * M + m] = value;
    }
}


int Chubarov_Mul(std::size_t N,
                 std::size_t M,
                 std::size_t L, 
                 float* output,
                 const float* first,
                 const float* second) {
    float* d_first  = nullptr;
    float* d_second = nullptr;
    float* d_output = nullptr;

    std::size_t first_size  = N * L * sizeof(float);
    std::size_t second_size = L * M * sizeof(float);
    std::size_t output_size = N * M * sizeof(float);

    cudaMalloc(&d_first,   first_size);
    cudaMalloc(&d_second, second_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_first,   first,  first_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_second, second, second_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    MulKernel<<<gridDim, blockDim>>>(N, M, L, d_output, d_first, d_second);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_first);
    cudaFree(d_second);
    cudaFree(d_output);

    return 0;
}
