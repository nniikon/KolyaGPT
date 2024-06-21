#include "../include/MLP.h"

#include <iostream>

// Layer
Layer::Layer(std::size_t rows, std::size_t cols) 
        : output_(rows, cols) {

}


std::size_t Layer::GetOutputRows() const { return output_.GetRows(); }
std::size_t Layer::GetOutputCols() const { return output_.GetCols(); }


// InputLayer
InputLayer::InputLayer(std::size_t n_inputs, std::size_t n_examples)
        : Layer      (n_examples, n_inputs),
          n_inputs_  (n_inputs), 
          n_examples_(n_examples) {
}


void InputLayer::SetValue(std::size_t example, std::size_t input, float value) {
    output_.SetValue(example, input, value);
}


std::size_t InputLayer::GetCols() const { return n_inputs_; }
std::size_t InputLayer::GetRows() const { return n_examples_; }
SmartMatrix* InputLayer::GetOutput() { return &output_; }


void InputLayer::ResetGrads() {
    output_.ResetGrad();
}


// MiddleLayer
MiddleLayer::MiddleLayer(Layer* input_layer, std::size_t n_outputs)
    : Layer         (input_layer->GetOutputRows(), n_outputs),
      input_layer_  (input_layer),
      n_input_rows_ (input_layer->GetOutputRows()),
      n_input_cols_ (input_layer->GetOutputCols()),
      n_output_cols_(n_outputs),
      weights_      (n_input_cols_, n_output_cols_),
      biases_       (1, n_output_cols_),
      unbiased_output_(n_input_rows_, n_output_cols_) {

    SetNormalRand();
}


void MiddleLayer::SetNormalRand() {
    weights_.SetMatrixNormRand();
    biases_ .SetMatrixNormRand();
}


void MiddleLayer::ResetGrads() {
    weights_.ResetGrad();
    biases_.ResetGrad();
    unbiased_output_.ResetGrad();
    output_.ResetGrad();
}


void MiddleLayer::Eval() {
    unbiased_output_.Mul(input_layer_->GetOutput(), &weights_);
    output_.AddVectorToMatrix(&unbiased_output_, &biases_);
}

void MiddleLayer::Backpropagate(float step) {
    weights_.AdjustValues(step);
    biases_ .AdjustValues(step);
}

SmartMatrix* MiddleLayer::GetOutput() { return &output_; }


// OutputLayer
OutputLayer::OutputLayer(Layer* input_layer, std::size_t n_outputs) 
    : MiddleLayer(input_layer, n_outputs),
      loss_(1, 1),
      expected_output_(output_.GetRows(), output_.GetCols()),
      norm_output_    (output_.GetRows(), output_.GetCols()){
}

void OutputLayer::SetExpectedValue(std::size_t example, std::size_t output, float value) {
    expected_output_.SetValue(example, output, value);
}

float OutputLayer::EvalLoss() {
    Eval();
    norm_output_.Sigm(&output_);
    loss_.Loss(&norm_output_, &expected_output_);
    loss_.EvalGrad();

    return loss_.GetValue(0, 0);
}


void OutputLayer::Dump() {
    loss_.Dump();
}


void OutputLayer::ResetGrads() {
    weights_.ResetGrad();
    biases_.ResetGrad();
    unbiased_output_.ResetGrad();
    output_.ResetGrad();
    norm_output_.ResetGrad();
    loss_.ResetGrad();
    expected_output_.ResetGrad();
}


float OutputLayer::GetNormOutput(std::size_t example, std::size_t output) {
    return norm_output_.GetValue(example, output);
}
