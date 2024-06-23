#include "../include/smart_matrix.h"

#include <assert.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

SmartMatrix::SmartMatrix(std::size_t n_rows, std::size_t n_cols)
    : values_(nullptr), grads_(nullptr),
      n_rows_(n_rows), n_cols_(n_cols), n_elems_(n_rows * n_cols) {

    values_ = new float[n_rows * n_cols]{};
    grads_  = new float[n_rows * n_cols]{};
    // FIXME: throw?
}


SmartMatrix::~SmartMatrix() {
    delete[] values_;
    delete[]  grads_;
}


float SmartMatrix::GetValue(std::size_t row, std::size_t col) const {
    return values_[row * n_cols_ + col];
}


float SmartMatrix::GetGrad(std::size_t row, std::size_t col) const {
    return grads_[row * n_cols_ + col];
}


std::size_t SmartMatrix::GetRows() const { return n_rows_; }
std::size_t SmartMatrix::GetCols() const { return n_cols_; }


void SmartMatrix::SetMatrixNormRand() {
    // https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    std::random_device rd;
    std::mt19937 gen(rd());

    const float mean = 0.0f;
    const float dispersion = 1.0f;
    std::normal_distribution<float> dis(mean, dispersion);

    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] = dis(gen);
    }
}


void SmartMatrix::SetMatrixValue(float value) {
    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] = value;
    }
}


void SmartMatrix::SetMatrixGrad(float value) {
    for (std::size_t i = 0; i < n_elems_; i++) {
        grads_[i] = value;
    }
}


void SmartMatrix::SetValue(std::size_t row, std::size_t col, float value) {
    values_[row * n_cols_ + col] = value;
}


void SmartMatrix::SetGrad(std::size_t row, std::size_t col, float value) {
    grads_[row * n_cols_ + col] = value;
}


void SmartMatrix::AddGrad(std::size_t row, std::size_t col, float value) {
    grads_[row * n_cols_ + col] += value;
}


void SmartMatrix::SetBinaryFamily(SmartMatrix* first, SmartMatrix* second,
                                  OperationType type_first, OperationType type_second) {
    first ->parent_ = this;
    second->parent_ = this;
    first ->parent_oper_ = type_first;
    second->parent_oper_ = type_second;
    first ->sibling_ = second;
    second->sibling_ = first;
    child1_ = first;
    child2_ = second;
}


void SmartMatrix::SetBinaryFamily(SmartMatrix* first, SmartMatrix* second,
                                  OperationType type) {
    SetBinaryFamily(first, second, type, type);
}


void SmartMatrix::SetUnaryFamily(SmartMatrix* first, OperationType type) {
    first ->parent_ = this;
    first ->parent_oper_ = type;
    child1_ = first;
    child2_ = nullptr;
}


void SmartMatrix::Loss(SmartMatrix* src, SmartMatrix* ref) {
    // FIXME: throw if matrices are differently sized

    assert(src->GetRows() == ref->GetRows());
    assert(src->GetCols() == ref->GetCols());
    assert(n_elems_ == 1);

    float loss = 0.0f;
    for (std::size_t i = 0; i < src->n_elems_; i++) {
        float diff = src->values_[i] - ref->values_[i];
        loss += diff * diff;
    }
    values_[0] = loss;

    SetBinaryFamily(src, ref, OperationType::LossSrc, OperationType::LossRef);
}


void SmartMatrix::AddVectorToMatrix (SmartMatrix* matrix, SmartMatrix* vector) {
    assert(n_rows_ == matrix->GetRows());
    assert(n_cols_ == matrix->GetCols());
    assert(matrix->GetCols() == vector->GetCols());
    assert(vector->GetRows() == 1);

    for (std::size_t i = 0; i < n_rows_; i++) {
        for (std::size_t j = 0; j < n_cols_; j++) {
            float new_value = matrix->GetValue(i, j) + vector->GetValue(0, j);
            SetValue(i, j, new_value);
        }
    }

    SetBinaryFamily(matrix, vector, OperationType::AddMatrix, OperationType::AddVector);
}



void SmartMatrix::Add(SmartMatrix* first, SmartMatrix* second) {
    // FIXME: throw if matrices are differently sized

    assert(n_rows_ == first->GetRows() && n_rows_ == second->GetRows());
    assert(n_cols_ == first->GetCols() && n_cols_ == second->GetCols());

    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] = first->values_[i] + second->values_[i];
    }

    SetBinaryFamily(first, second, OperationType::Add);
}


void SmartMatrix::Sub(SmartMatrix* first, SmartMatrix* second) {
    // FIXME: throw if matrices are differently sized

    assert(n_rows_ == first->GetRows() && n_rows_ == second->GetRows());
    assert(n_cols_ == first->GetCols() && n_cols_ == second->GetCols());

    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] = first->values_[i] - second->values_[i];
    }

    SetBinaryFamily(first, second, OperationType::LSub, OperationType::RSub);
}


void SmartMatrix::Mul(SmartMatrix* first, SmartMatrix* second) {
    // FIXME: throw if matrices are differently sized
    assert(n_rows_ == first->GetRows() && n_cols_ == second->GetCols());
    assert(first->GetCols() == second->GetRows());

    // Local notation: (NxL) * (LxM) = (NxM)
    std::size_t N = n_rows_;
    std::size_t M = n_cols_;
    std::size_t L = first->GetCols();

    // FIXME: optimize
    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            float dot_product = 0.0f;

            for (std::size_t l = 0; l < L; l++) {
                dot_product += first->GetValue(n, l) * second->GetValue(l, m);
            }

            SetValue(n, m, dot_product);
        }
    }

    SetBinaryFamily(first, second, OperationType::LMul, OperationType::RMul);
}


void SmartMatrix::Sigm(SmartMatrix* first) {
    // FIXME: throw if matrices are differently sized
    assert(n_rows_ == first->GetRows());
    assert(n_cols_ == first->GetCols());

    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] = 1 / (1 + expf(-first->values_[i]));
    }

    SetUnaryFamily(first, OperationType::Sigm);
}


void SmartMatrix::DumpMatrix_(std::ofstream& out) const {
    out << "Node" << this << " [label=\"{";

    out << "Values:|";
    for (std::size_t i = 0; i < n_rows_; ++i) {
        for (std::size_t j = 0; j < n_cols_; ++j) {
            out << values_[i * n_cols_ + j];
            if (j < n_cols_ - 1) {
                out << ", ";
            }
        }
        if (i < n_rows_ - 1) {
            out << "|";
        }
    }

    out << "} | {";

    out << "Grads:|";
    for (std::size_t i = 0; i < n_rows_; ++i) {
        for (std::size_t j = 0; j < n_cols_; ++j) {
            out << grads_[i * n_cols_ + j];
            if (j < n_cols_ - 1) {
                out << ", ";
            }
        }
        if (i < n_rows_ - 1) {
            out << "|";
        }
    }

    out << "}\"];\n";
}


void SmartMatrix::AdjustValues(float step) {
    for (std::size_t i = 0; i < n_elems_; i++) {
        values_[i] -= step * grads_[i];
    }
}


void SmartMatrix::ResetGrad() {
    for (std::size_t i = 0; i < n_elems_; i++) {
        grads_[i] = 0.0f;
    }
}


void SmartMatrix::EvalGrad() {
    SetMatrixGrad(1.0f); // dx/dx is 1 by definition

    if (child1_) { child1_->EvalGradRecursive_(); }
    if (child2_) { child2_->EvalGradRecursive_(); }
}


void SmartMatrix::EvalGradRecursive_() {
    assert(parent_ != nullptr);

    switch(parent_oper_) {
        case OperationType::RSub:
            EvalGradRSub_();
            break;
        case OperationType::AddMatrix:
        case OperationType::LSub:
        case OperationType::Add:
            EvalGradAddMatrixLSubAdd_();
            break;
        case OperationType::LMul:
            EvalGradLMul_();
            break;
        case OperationType::RMul:
            EvalGradRMul_();
            break;
        case OperationType::Sigm:
            EvalGradSigm_();
            break;
        case OperationType::LossSrc:
            EvalGradLossSrc_();
            break;
        case OperationType::AddVector:
            EvalGradAddVector_();
            break;
        case OperationType::LossRef:
            // Not needed
            break;
        case OperationType::None:
        default:
            assert(0);
    }

    if (child1_) { child1_->EvalGradRecursive_(); }
    if (child2_) { child2_->EvalGradRecursive_(); }
}

void SmartMatrix::EvalGradRSub_() {
    for (std::size_t i = 0; i < n_elems_; i++) {
        grads_[i] += -parent_->grads_[i];
    }
}

void SmartMatrix::EvalGradAddMatrixLSubAdd_() {
    for (std::size_t i = 0; i < n_elems_; i++) {
        grads_[i] += parent_->grads_[i];
    }
}

void SmartMatrix::EvalGradLMul_() {
    std::size_t N = n_rows_;
    std::size_t M = parent_->GetCols();
    std::size_t L = n_cols_;

    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            for (std::size_t l = 0; l < L; l++) {
                float local_grad = sibling_->GetValue(l, m);
                float parent_grad = parent_->GetGrad(n, m);
                AddGrad(n, l, local_grad * parent_grad);
            }
        }
    }
}

void SmartMatrix::EvalGradRMul_() {
    std::size_t N = parent_->GetRows();
    std::size_t M = n_cols_;
    std::size_t L = n_rows_;

    for (std::size_t n = 0; n < N; n++) {
        for (std::size_t m = 0; m < M; m++) {
            for (std::size_t l = 0; l < L; l++) {
                float local_grad = sibling_->GetValue(n, l);
                float parent_grad = parent_->GetGrad(n, m);
                AddGrad(l, m, local_grad * parent_grad);
            }
        }
    }
}

void SmartMatrix::EvalGradSigm_() {
    for (std::size_t i = 0; i < n_elems_; i++) {
        float local_grad = parent_->values_[i] * (1 - parent_->values_[i]);
        grads_[i] += parent_->grads_[i] * local_grad;
    }
}

void SmartMatrix::EvalGradLossSrc_() {
    for (std::size_t i = 0; i < n_elems_; i++) {
        float local_grad = 2 * (values_[i] - sibling_->values_[i]);
        grads_[i] += parent_->grads_[0] * local_grad;
    }
}

void SmartMatrix::EvalGradAddVector_() {
    for (std::size_t i = 0; i < n_cols_; i++) {
        for (std::size_t j = 0; j < parent_->n_rows_; j++) {
            grads_[i] += parent_->GetGrad(j, i);
        }
    }
}


void SmartMatrix::Dump() const {

    std::string   file_name = "graph.dot";
    std::string output_name = "smart_matrix_dump.png";

    std::ofstream out(file_name);
    // Exception??
    out << "digraph G {\n";
    out << "\tnode [shape=record];\n";

    DumpRecursive_(false, out);

    out << "}\n";
    out.close();

    std::string compile_cmd = "dot -Tpng " + file_name + " -o " + output_name;
    system(compile_cmd.c_str());
}


void SmartMatrix::DumpRecursive_(bool isSibling, std::ofstream& out) const {
    if (child1_) { child1_->DumpRecursive_(false, out); }
    if (child2_) { child2_->DumpRecursive_(true , out); } 

    DumpMatrix_(out);
    if (parent_) {
        const char* op_str = nullptr;

        switch(parent_oper_) {
            case OperationType::AddMatrix:
            case OperationType::AddVector:
            case OperationType::Add:        op_str = "+";       break;
            case OperationType::RMul:
            case OperationType::LMul:       op_str = "*";       break;
            case OperationType::RSub:
            case OperationType::LSub:       op_str = "-";       break;
            case OperationType::Sigm:       op_str = "sigm";    break;
            case OperationType::LossSrc:
            case OperationType::LossRef:    op_str = "loss";    break;

            case OperationType::None:
            default:
                assert(0);
                op_str = "NONE";
                break;
        }

        if (!isSibling) {
            out << "\top" << parent_ << " [label=\" " << op_str << "\"];\n"; 
        }

        out << "\tNode"   << this    << " -> " << "op" << parent_ << ";\n";
        out << "\top" << parent_ << " -> Node" <<         parent_ << ";\n";
    }
}
