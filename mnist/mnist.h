#ifndef MNIST_H_
#define MNIST_H_

#include "mnist_parser/mnist_parser.h"
#include "../include/MLP.h"

#include <cstdlib>
#include <vector>
#include <memory>

class Mnist {
    public:
        Mnist(const char* train_images_path,
              const char* train_labels_path,
              const char* weights_folder_path,
              std::size_t n_hidden_layers = 2,
              std::size_t n_hidden_layer_neurons = 12);

        void LoadWeights();
        void SaveWeights();

        float Eval();
        void Backpropagate(float step);

        void EvalImage(float* input);

    private:
        MnistParser mnist_parser_;
        MnistImages mnist_images_;
        MnistLabels mnist_labels_;

        const char* weights_folder_path_;

        uint8_t* images_buffer_;
        uint8_t* labels_buffer_;

        const std::size_t n_examples_;
        const std::size_t n_input_neurons_;
        const std::size_t n_hidden_layers_;
        const std::size_t n_hidden_layer_neurons_;
        const std::size_t n_output_neurons_       = 10;

        std::unique_ptr<InputLayer>         input_layer_;
        std::vector<MiddleLayer>            middle_layers_;
        std::unique_ptr<OutputLayerDiscret> output_layer_;

        SmartMatrix  input_test_vector_;
        SmartMatrix  output_test_vector_;

        std::vector<std::string> middle_layers_names_;
        std::string              output_layer_name_;


        void Dump();
};

#endif // MNIST_H_
