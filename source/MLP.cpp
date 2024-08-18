#include "../include/MLP.h"

#include <iostream>
#include <assert.h>

//================================ Layer ======================================

Layer::Layer(std::size_t rows, std::size_t cols, Layer* input_layer)
        : input_layer_(input_layer),
          output_(rows, cols) {
}


Layer::~Layer() {
    // No need to delete input_layer_ as it is not owned by this class
}


Layer::Layer(const Layer& other)
    : input_layer_(other.input_layer_),
      output_(other.output_) {
}


Layer& Layer::operator=(const Layer& other) {
    if (this == &other) return *this;

    input_layer_ = other.input_layer_;
    output_      = other.output_;

    return *this;
}


Layer::Layer(Layer&& other) 
    : input_layer_(other.input_layer_),
      output_(std::move(other.output_)) {

    other.input_layer_ = nullptr;
}


Layer& Layer::operator=(Layer&& other) {
    if (this == &other) return *this;

    input_layer_ = other.input_layer_;
    output_ = std::move(other.output_);

    other.input_layer_ = nullptr;

    return *this;
}


void Layer::SetInputLayer(Layer* layer) {input_layer_ = layer; }

Layer*      Layer::GetInputLayer() const { return input_layer_; }
std::size_t Layer::GetOutputRows() const { return output_.GetRows(); }
std::size_t Layer::GetOutputCols() const { return output_.GetCols(); }


//================================ InputLayer =================================

InputLayer::InputLayer(std::size_t n_inputs, std::size_t n_examples)
        : Layer      (n_examples, n_inputs, nullptr),
          n_inputs_  (n_inputs), 
          n_examples_(n_examples) {
}


InputLayer::~InputLayer() {
}


InputLayer::InputLayer(const InputLayer& other)
    : Layer      (other),
      n_inputs_  (other.n_inputs_),
      n_examples_(other.n_examples_) {
}


InputLayer& InputLayer::operator=(const InputLayer& other) {
    if (this == &other) return *this;

    assert(n_inputs_   == other.n_inputs_);
    assert(n_examples_ == other.n_examples_);

    Layer::operator=(other);

    return *this;
}


InputLayer::InputLayer(InputLayer&& other) 
    : Layer(std::move(other)),
      n_inputs_  (other.n_inputs_),
      n_examples_(other.n_examples_) {
}


InputLayer& InputLayer::operator=(InputLayer&& other) {
    if (this == &other) return *this;

    assert(n_inputs_   == other.n_inputs_);
    assert(n_examples_ == other.n_examples_);

    Layer::operator=(std::move(other));

    return *this;
}


void InputLayer::SetValue(std::size_t example, std::size_t input, float value) {
    output_.SetValue(example, input, value);
}


std::size_t InputLayer::GetCols() const { return n_inputs_;   }
std::size_t InputLayer::GetRows() const { return n_examples_; }


SmartMatrix* InputLayer::GetOutput() { return &output_; }


void InputLayer::ResetGrads() {
    output_.ResetGrad();
}

void InputLayer::EvalRecursive()                    { /* nothing here */}
void InputLayer::ResetGradsRecursive()              { ResetGrads();     }
void InputLayer::BackpropagateRecursive(float step) { /* nothing here */}

//================================ MiddleLayer ================================

MiddleLayer::MiddleLayer(Layer* input_layer, std::size_t n_outputs)
    : Layer         (input_layer->GetOutputRows(), n_outputs, input_layer),
      n_input_rows_ (input_layer->GetOutputRows()),
      n_input_cols_ (input_layer->GetOutputCols()),
      n_output_cols_(n_outputs),
      weights_        (n_input_cols_, n_output_cols_),
      biases_         (1            , n_output_cols_),
      unbiased_output_(n_input_rows_, n_output_cols_),
      norm_output_    (n_input_rows_, n_output_cols_) {

    SetNormalRand();
}


MiddleLayer::~MiddleLayer() {
}

MiddleLayer::MiddleLayer(const MiddleLayer& other)
    : Layer           (other),
      n_input_rows_   (other.n_input_rows_),
      n_input_cols_   (other.n_input_cols_),
      n_output_cols_  (other.n_output_cols_),
      weights_        (other.weights_),
      biases_         (other.biases_),
      unbiased_output_(other.unbiased_output_),
      norm_output_    (other.norm_output_) {
}

MiddleLayer& MiddleLayer::operator=(const MiddleLayer& other) {
    if (this == &other) return *this;

    assert(n_input_rows_  == other.n_input_rows_);
    assert(n_input_cols_  == other.n_input_cols_);
    assert(n_output_cols_ == other.n_output_cols_);

    Layer::operator=(other);

    weights_         = other.weights_;
    biases_          = other.biases_;
    unbiased_output_ = other.unbiased_output_;
    norm_output_     = other.norm_output_;

    return *this;
}


MiddleLayer::MiddleLayer(MiddleLayer&& other) 
    : Layer(std::move(other)),
      n_input_rows_     (other.n_input_rows_),
      n_input_cols_     (other.n_input_cols_),
      n_output_cols_    (other.n_output_cols_),
      weights_          (std::move(other.weights_)),
      biases_           (std::move(other.biases_)),
      unbiased_output_  (std::move(other.unbiased_output_)),
      norm_output_      (std::move(other.norm_output_)) {
}


MiddleLayer& MiddleLayer::operator=(MiddleLayer&& other) {
    if (this == &other) return *this;

    Layer::operator=(std::move(other));

    assert(n_input_rows_  == other.n_input_rows_);
    assert(n_input_cols_  == other.n_input_cols_);
    assert(n_output_cols_ == other.n_output_cols_);

    weights_         = std::move(other.weights_);
    biases_          = std::move(other.biases_);
    unbiased_output_ = std::move(other.unbiased_output_);
    norm_output_     = std::move(other.norm_output_);

    return *this;
}


void MiddleLayer::SaveParamsToFile(const char* file_name) {
    assert(file_name);

    const float* weights = weights_.GetValues();
    const float* biases  = biases_ .GetValues();

    size_t n_weights = weights_.GetRows() * weights_.GetCols();
    size_t n_biases  =  biases_.GetRows() *  biases_.GetCols();

    std::ofstream ofs(file_name, std::ios::binary);
    assert(ofs);

    ofs.write(reinterpret_cast<const char*>(&n_weights), sizeof(n_weights));
    ofs.write(reinterpret_cast<const char*>(weights), n_weights * sizeof(float));

    ofs.write(reinterpret_cast<const char*>(&n_biases), sizeof(n_biases));
    ofs.write(reinterpret_cast<const char*>(biases), n_biases * sizeof(float));

    ofs.close();
}


void MiddleLayer::LoadParamsFromFile(const char* file_name) {
    assert(file_name);

    std::ifstream ifs(file_name, std::ios::binary);
    assert(ifs);

    size_t n_weights = 0;
    size_t n_biases  = 0;

    ifs.read(reinterpret_cast<char*>(&n_weights), sizeof(n_weights));
    float* weights = new float[n_weights];
    ifs.read(reinterpret_cast<char*>(weights), n_weights * sizeof(float));

    ifs.read(reinterpret_cast<char*>(&n_biases), sizeof(n_biases));
    float* biases = new float[n_biases];
    ifs.read(reinterpret_cast<char*>(biases), n_biases * sizeof(float));

    ifs.close();

    weights_.SetValues(weights);
    biases_ .SetValues(biases);
}


void MiddleLayer::SetNormalRand() {
    weights_.SetMatrixNormRand();
    biases_ .SetMatrixNormRand();
}


void MiddleLayer::ResetGrads() {
    weights_        .ResetGrad();
    biases_         .ResetGrad();
    unbiased_output_.ResetGrad();
    output_         .ResetGrad();
    norm_output_    .ResetGrad();
}


void MiddleLayer::Eval() {
    unbiased_output_.Mul(input_layer_->GetOutput(), &weights_);
    output_.AddVectorToMatrix(&unbiased_output_, &biases_);
    norm_output_.Sigm(&output_);
}


void MiddleLayer::Backpropagate(float step) {
    weights_.AdjustValues(step);
    biases_ .AdjustValues(step);
}


SmartMatrix* MiddleLayer::GetOutput() { return &norm_output_; }

void MiddleLayer::EvalRecursive() {
    assert(input_layer_);

    input_layer_->EvalRecursive();
    Eval();
}


void MiddleLayer::ResetGradsRecursive() {
    assert(input_layer_);

    input_layer_->ResetGradsRecursive();
    ResetGrads();
}


void MiddleLayer::BackpropagateRecursive(float step) {
    assert(input_layer_);

    input_layer_->BackpropagateRecursive(step);
    Backpropagate(step);
}

//================================ OutputLayer ================================

OutputLayer::OutputLayer(Layer* input_layer, std::size_t n_outputs) 
    : MiddleLayer(input_layer, n_outputs),
      loss_(1, 1),
      expected_output_(output_.GetRows(), output_.GetCols()) {
}


OutputLayer::~OutputLayer() {
}


OutputLayer::OutputLayer(const OutputLayer& other)
    : MiddleLayer        (other),
      loss_              (other.loss_),
      expected_output_   (other.expected_output_) {
}


OutputLayer& OutputLayer::operator=(const OutputLayer& other) {
    if (this == &other) return *this;

    MiddleLayer::operator=(other);

    loss_            = other.loss_;
    expected_output_ = other.expected_output_;

    return *this;
}


OutputLayer::OutputLayer(OutputLayer&& other) 
    : MiddleLayer        (std::move(other)),
      loss_              (std::move(other.loss_)),
      expected_output_   (std::move(other.expected_output_)) {
}


OutputLayer& OutputLayer::operator=(OutputLayer&& other) {
    if (this == &other) return *this;

    MiddleLayer::operator=(std::move(other));

    loss_               = std::move(other.loss_);
    expected_output_    = std::move(other.expected_output_);

    return *this;
}


void OutputLayer::SetExpectedValue(std::size_t example, std::size_t output, float value) {
    expected_output_.SetValue(example, output, value);
}


float OutputLayer::GetExpectedValue(std::size_t example, std::size_t output) {
    return expected_output_.GetValue(example, output);
}


void OutputLayer::Eval() {
    unbiased_output_.Mul(input_layer_->GetOutput(), &weights_);
    output_.AddVectorToMatrix(&unbiased_output_, &biases_);
    norm_output_.Softmax(&output_);
}


float OutputLayer::EvalLoss() {
    Eval();
    loss_.SquaredErrorLoss(&norm_output_, &expected_output_);
    loss_.EvalGrad();

    return loss_.GetValue(0, 0);
}


void OutputLayer::Dump() {
    loss_.Dump();
}


void OutputLayer::ResetGrads() {
    weights_        .ResetGrad();
    biases_         .ResetGrad();
    unbiased_output_.ResetGrad();
    output_         .ResetGrad();
    norm_output_    .ResetGrad();
    loss_           .ResetGrad();
    expected_output_.ResetGrad();
}


float OutputLayer::GetLoss() const { return loss_.GetValue(0, 0); }

float OutputLayer::GetNormOutput(std::size_t example, std::size_t output) {
    return norm_output_.GetValue(example, output);
}


void OutputLayer::EvalRecursive() {
    assert(input_layer_);

    input_layer_->EvalRecursive();
    EvalLoss();
}


void OutputLayer::ResetGradsRecursive() {
    assert(input_layer_);

    input_layer_->ResetGradsRecursive();
    ResetGrads();
}


void OutputLayer::BackpropagateRecursive(float step) {
    assert(input_layer_);

    input_layer_->BackpropagateRecursive(step);
    Backpropagate(step);
}
