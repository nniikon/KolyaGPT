#ifndef CHUBAROV_H_
#define CHUBAROV_H_

#include <cstddef>

int Chubarov_EvalGradLMul(std::size_t N,
                          std::size_t M,
                          std::size_t L, 
                          float* grads,
                          const float* sibling_values,
                          const float* parent_grads);

int Chubarov_EvalGradRMul(std::size_t N,
                          std::size_t M,
                          std::size_t L, 
                          float* grads,
                          const float* sibling_values,
                          const float* parent_grads);

int Chubarov_Mul(std::size_t N,
                 std::size_t M,
                 std::size_t L, 
                 float* output,
                 const float* first,
                 const float* second);

#endif // CHUBAROV_H_
