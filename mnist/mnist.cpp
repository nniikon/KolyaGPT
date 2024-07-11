#include "mnist.h"

#include <assert.h>
#include <iostream>

void Mnist::Dump() {
    std::cout << "InputLayer: " << input_layer_.get() << "\n";
    std::cout << "\t Parent: " << input_layer_->GetInputLayer() << "\n";
    for (std::size_t i = 0; i < n_hidden_layers_; i++) {
        std::cout << "MiddleLayer" << i << ": " << &middle_layers_[i] << "\n";
        std::cout << "\t Parent: " << middle_layers_[i].GetInputLayer() << "\n";
    }
    std::cout << "OutputLayer: " << output_layer_.get() << "\n";
    std::cout << "\t Parent: " << output_layer_->GetInputLayer() << "\n";
}


Mnist::Mnist(const char* train_images_path,
             const char* train_labels_path,
             const char* weights_folder_path,
             std::size_t n_hidden_layers,
             std::size_t n_hidden_layer_neurons)
    : mnist_parser_(train_images_path, train_labels_path),
      mnist_images_(mnist_parser_.GetMnistImages()),
      mnist_labels_(mnist_parser_.GetMnistLabels()),
      weights_folder_path_(weights_folder_path),
      n_examples_(static_cast<std::size_t>(mnist_labels_.n_labels)),
      n_input_neurons_(mnist_images_.n_cols * mnist_images_.n_rows),
      n_hidden_layers_(n_hidden_layers),
      n_hidden_layer_neurons_(n_hidden_layer_neurons),
      input_test_vector_ (1, n_input_neurons_),
      output_test_vector_(1, n_output_neurons_) {

    assert(mnist_labels_.n_labels == mnist_images_.n_images);

    input_layer_ = std::make_unique<InputLayer>(n_input_neurons_, n_examples_);

    middle_layers_.reserve(n_hidden_layers_);
    middle_layers_.emplace_back(MiddleLayer(input_layer_.get(), n_hidden_layer_neurons_)) ;
    for (std::size_t i = 1; i < n_hidden_layers_; i++) {
        middle_layers_.emplace_back(MiddleLayer(&middle_layers_[i - 1], n_hidden_layer_neurons_));
    }

    output_layer_ = std::make_unique<OutputLayer>(&middle_layers_[n_hidden_layers_ - 1], n_output_neurons_);

    // Dump();

    images_buffer_ = mnist_images_.buffer;
    labels_buffer_ = mnist_labels_.buffer;

    for (std::size_t example = 0; example < n_examples_; example++) {
        for (std::size_t neuron = 0; neuron < n_input_neurons_; neuron++) {
            float value = (float)images_buffer_[example * n_input_neurons_ + neuron]/256.0f;
            input_layer_->SetValue(example, neuron, value);
        }
        output_layer_->SetExpectedValue(example, labels_buffer_[example], 1.0f);
    }

    // FIXME: fix copypaste
    middle_layers_names_.reserve(n_hidden_layers);
    for (std::size_t i = 0; i < n_hidden_layers_; i++) {
        middle_layers_names_.emplace_back(
                std::string(weights_folder_path_) + "/" + std::string("middle")
                                                  + std::to_string(i + 1) + ".data");
    }
    output_layer_name_ = std::string(weights_folder_path_) + "/" + std::string("output") + ".data";
}


void Mnist::LoadWeights() {
    for (std::size_t i = 0; i < n_hidden_layers_; i++) {
        std::cout << "Trying to load from: " << middle_layers_names_[i].c_str() << std::endl;
        middle_layers_[i].LoadParamsFromFile(middle_layers_names_[i].c_str());
    }
    std::cout << "Trying to load from: " << output_layer_name_.c_str() << std::endl;
    output_layer_->LoadParamsFromFile(output_layer_name_.c_str());
}


void Mnist::SaveWeights() {
    for (std::size_t i = 0; i < n_hidden_layers_; i++) {
        middle_layers_[i].SaveParamsToFile(middle_layers_names_[i].c_str());
    }
    output_layer_->SaveParamsToFile(output_layer_name_.c_str());
}


float Mnist::Eval() {
    output_layer_->ResetGradsRecursive();
    output_layer_->EvalRecursive();
    return output_layer_->GetLoss();
}


void Mnist::Backpropagate(float step) {
    output_layer_->BackpropagateRecursive(step);
}


void Mnist::EvalImage(float* input) {
    assert(input);

    for (std::size_t i = 0; i < n_input_neurons_; i++) {
        input_layer_->SetValue(0, i, input[i]);
    }

    Eval();

    for (std::size_t i = 0; i < n_output_neurons_; i++) {
        std::cout << i << ": " << output_layer_->GetNormOutput(0, i) * 100.0f << "%\n";
    }
}
