#include "../chubarov.h"

int Chubarov_EvalGradLMul(std::size_t N,
                          std::size_t M,
                          std::size_t L, 
                          float* grads,
                          const float* sibling_values,
                          const float* parent_grads) {

    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            float parent_grad = parent_grads[n * M + m];
            for (std::size_t l = 0; l < L; l++) {
                float local_grad = sibling_values[l * M + m];
                grads[n * L + l] += local_grad * parent_grad;
            }
        }
    }

    return 0;
}

int Chubarov_EvalGradRMul(std::size_t N,
                          std::size_t M,
                          std::size_t L, 
                          float* grads,
                          const float* sibling_values,
                          const float* parent_grads) {

    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            for (std::size_t l = 0; l < L; l++) {
                float local_grad = sibling_values[n * L + l];
                float parent_grad = parent_grads[n * M + m];
                grads[l * M + m] += local_grad * parent_grad;
            }
        }
    }

    return 0;
}

int Chubarov_Mul(std::size_t N,
                 std::size_t M,
                 std::size_t L, 
                 float* output,
                 const float* first,
                 const float* second) {

    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            float value = 0.0f;

            for (std::size_t l = 0; l < L; l++) {
                value += first[n * L + l] * second[l * M + m];
            }

            output[n * M + m] = value;
        }
    }

    return 0;
}
