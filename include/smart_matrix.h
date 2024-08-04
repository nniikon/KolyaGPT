#ifndef SMART_MATRIX_H_
#define SMART_MATRIX_H_

#include <cstddef>
#include <fstream>
#include "../chubarov_lib/chubarov.h"

class SmartMatrix {

    public:
        SmartMatrix(std::size_t n_rows, std::size_t n_cols);
        SmartMatrix(const SmartMatrix& other);
        SmartMatrix(SmartMatrix&& other);
        SmartMatrix& operator=(const SmartMatrix& other);
        SmartMatrix& operator=(SmartMatrix&& other);
        ~SmartMatrix();

        void Add               (SmartMatrix* first,  SmartMatrix* second);
        void AddVectorToMatrix (SmartMatrix* matrix, SmartMatrix* vector);
        void Sub               (SmartMatrix* first,  SmartMatrix* second);
        void Mul               (SmartMatrix* first,  SmartMatrix* second);
        void Loss              (SmartMatrix* src,    SmartMatrix* ref);
        void Sigm              (SmartMatrix* first);
        void Softmax           (SmartMatrix* matrix);

        float GetValue(std::size_t row, std::size_t col) const;
        float GetGrad (std::size_t row, std::size_t col) const;
        std::size_t GetRows() const;
        std::size_t GetCols() const;

        void SetMatrixValue(float value);
        void SetMatrixNormRand();
        void SetMatrixGrad(float value);
        void SetValue(std::size_t row, std::size_t col, float value);
        void SetGrad (std::size_t row, std::size_t col, float value);
        void AddGrad (std::size_t row, std::size_t col, float value);

        const float* GetValues() const;
        void SetValues(float* values);

        void EvalGrad();
        void ResetGrad();
        void AdjustValues(float step);

        void Dump() const;

    private:
        enum class OperationType {
            None,
            Add,
            AddMatrix,
            AddVector,
            LSub,
            RSub,
            LMul,
            RMul,
            Sigm,
            Softmax,
            LossSrc,
            LossRef,
        };

        float* values_;
        float* grads_;
        const std::size_t n_rows_;
        const std::size_t n_cols_;
        const std::size_t n_elems_;
        OperationType parent_oper_;
        SmartMatrix* sibling_;
        SmartMatrix* parent_;
        SmartMatrix* child1_;
        SmartMatrix* child2_;

        void SetBinaryFamily(SmartMatrix* first, SmartMatrix* second,
                             OperationType type_first, OperationType type_second);
        void SetBinaryFamily(SmartMatrix* first, SmartMatrix* second,
                             OperationType type);
        void SetUnaryFamily(SmartMatrix* first, OperationType type);

        void DumpMatrix_   (                std::ofstream& out) const;
        void DumpRecursive_(bool isSibling, std::ofstream& out) const;

        void EvalGradRecursive_();
        void EvalGradRSub_();
        void EvalGradAddMatrixLSubAdd_();
        void EvalGradLMul_();
        void EvalGradRMul_();
        void EvalGradSigm_();
        void EvalGradSoftmax_();
        void EvalGradLossSrc_();
        void EvalGradAddVector_();
};

#endif // SMART_MATRIX_H_

