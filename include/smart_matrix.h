#ifndef SMART_MATRIX_H_
#define SMART_MATRIX_H_

#include <cstddef>
#include <fstream>

class SmartMatrix {

    public:
        SmartMatrix(std::size_t n_rows, std::size_t n_cols);
        ~SmartMatrix();
        void Add (SmartMatrix* first, SmartMatrix* second);
        void Sub (SmartMatrix* first, SmartMatrix* second);
        void Mul (SmartMatrix* first, SmartMatrix* second);
        void Loss(SmartMatrix* src,   SmartMatrix* ref);
        void Sigm(SmartMatrix* first);

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

        void EvalGrad();
        void ResetGrad();
        void AdjustValues(float step);

        void Dump() const;

    private:
        enum class OperationType {
            None,
            Add,
            LSub,
            RSub,
            LMul,
            RMul,
            Sigm,
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

        float GetElem(float* data, std::size_t row, std::size_t col) const;
        void  SetElem(float* data, std::size_t row, std::size_t col, float value);

        void DumpMatrix_   (                std::ofstream& out) const;
        void DumpRecursive_(bool isSibling, std::ofstream& out) const;

        void EvalGradRecursive_();
};

#endif // SMART_MATRIX_H_

