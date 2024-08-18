#include "include/smart_matrix.h"
#include "include/MLP.h"
#include "mnist/mnist_parser/mnist_parser.h"

#include <iostream>
#include <assert.h>
#include <immintrin.h>
#include <iomanip>

void TestSmartMatrix();
void TestMLP();
void TestMnistParser();
void TestWriting();
void TestMnistLib();

void TrainMnist();

int main() {
    TestMLP();
}


void TestSmartMatrix() {
    const std::size_t kExampleInputs = 10;
    const std::size_t kInputSize     = 14;
    const std::size_t kOutputSize    = 15;

    SmartMatrix input_layers   (kExampleInputs, kInputSize);
    SmartMatrix weights        (kInputSize, kOutputSize);
    SmartMatrix unbiased_output(kExampleInputs, kOutputSize);
    SmartMatrix     norm_output(kExampleInputs, kOutputSize);
    SmartMatrix     prob_output(kExampleInputs, kOutputSize);
    SmartMatrix expected_output(kExampleInputs, kOutputSize);
    SmartMatrix            loss(1, 1);

    for (std::size_t j = 0; j < kExampleInputs; j++)
        for (std::size_t i = 0; i < kOutputSize; i++)
            expected_output.SetValue(j, i, i == j);

    input_layers.SetMatrixNormRand();
    weights     .SetMatrixNormRand();

    for (std::size_t i = 0; i < 100'000; i++) {
        weights.        ResetGrad();
        input_layers.   ResetGrad();
        unbiased_output.ResetGrad();
        norm_output.    ResetGrad();
        prob_output.    ResetGrad();
        loss.           ResetGrad();

        unbiased_output.Mul(&input_layers, &weights);
        prob_output.Softmax(&unbiased_output);
        loss.Loss          (&prob_output, &expected_output);

        loss.EvalGrad();

        if (i % 1000 == 0)
            std::cout << "Iteration " << i << ": loss = " << loss.GetValue(0, 0) << "\n"; 

        weights.AdjustValues(0.1f);
    }

    for (std::size_t j = 0; j < kExampleInputs; j++) {
        for (size_t i = 0; i < kOutputSize; i++) {
            std::cout << "Prob " << i << " = " << prob_output.GetValue(j, i) << "\tExpected = " << expected_output.GetValue(j, i) << "\n";
        }
        std::cout << std::endl;
    }

    loss.Dump();
}


void TestMLP() {
    const std::size_t kExamples      = 10;
    const std::size_t kInputNeurons  = 28;
    const std::size_t kMiddleNeurons = 12;
    const std::size_t kOutputNeurons = 10;

    InputLayer   input_layer (kInputNeurons,  kExamples);
    MiddleLayer middle_layer1(&input_layer,   kMiddleNeurons);
    MiddleLayer middle_layer2(&middle_layer1, kMiddleNeurons);
    OutputLayer output_layer (&middle_layer2, kOutputNeurons);

    for (std::size_t i = 0; i < kExamples; i++) {
        for (std::size_t j = 0; j < kInputNeurons; j++) {
            input_layer.SetValue(i, j, (float)(i + j) / (kExamples + kInputNeurons));
        }

        for (std::size_t j = 0; j < kOutputNeurons; j++) {
            output_layer.SetExpectedValue(i, j, i == j);
        }
    }
    middle_layer1.SetNormalRand();
    middle_layer2.SetNormalRand();
    output_layer.SetNormalRand();

    const float kStep = 0.0001f;
    const std::size_t kIterations = 1000000;
    for (std::size_t i = 0; i < kIterations; i++) {
        input_layer  .ResetGrads();
        middle_layer1.ResetGrads();
        middle_layer2.ResetGrads();
        output_layer .ResetGrads();

        middle_layer1.Eval();
        middle_layer2.Eval();
        float loss = output_layer.EvalLoss();

        std::cout << "Iteration " << i << ": loss = " << loss << "\n"; 

        //if (i == kIterations - 1 || i == 0)
        //    for (size_t l = 0; l < kExamples; l++) {
        //        for (size_t j = 0; j < kOutputNeurons; j++) {
        //            std::cout << output_layer.GetNormOutput(l, j) << "\tExpected = " << (float)(l+j) / (kExamples + kInputNeurons - 2) << "\n";
        //        }
        //    }

        middle_layer1.Backpropagate(kStep);
        middle_layer2.Backpropagate(kStep);
        output_layer .Backpropagate(kStep);
    }
    output_layer.Dump();
}


void TestMnistParser() {
    MnistParser mnist_parser("mnist/mnist_training_data/train-images.idx3-ubyte",
                             "mnist/mnist_training_data/train-labels.idx1-ubyte");

    MnistLabels mnist_labels = mnist_parser.GetMnistLabels();
    MnistImages mnist_images = mnist_parser.GetMnistImages();

    std::cout << "Images:\n\tn_images = " << mnist_images.n_images << "\n";
    std::cout << "\tgrid = " << mnist_images.n_cols << " x " << mnist_images.n_rows << "\n\n";

    std::cout << "Labels:\n\tn_labels = " << mnist_labels.n_labels<< "\n";
}


const char* middle_layer1_saveload = "mnist/mnist_weights/middle1.data";
const char* middle_layer2_saveload = "mnist/mnist_weights/middle2.data";
const char*        output_saveload = "mnist/mnist_weights/output.data";
void TrainMnist() {
    MnistParser mnist_parser("mnist/mnist_training_data/train-images.idx3-ubyte",
                             "mnist/mnist_training_data/train-labels.idx1-ubyte");

    MnistImages mnist_images = mnist_parser.GetMnistImages();
    MnistLabels mnist_labels = mnist_parser.GetMnistLabels();

    uint8_t* images_buffer = mnist_images.buffer;
    uint8_t* labels_buffer = mnist_labels.buffer;

    assert(mnist_labels.n_labels == mnist_images.n_images);

    // const std::size_t kExamples      = static_cast<std::size_t>(mnist_labels.n_labels);
    const std::size_t kExamples      = 2'0;
    const std::size_t kInputNeurons  = mnist_images.n_cols * mnist_images.n_rows;
    const std::size_t kMiddleNeurons = 16;
    const std::size_t kOutputNeurons = 10;

    InputLayer   input_layer (kInputNeurons,  kExamples);
    MiddleLayer middle_layer1(&input_layer,   kMiddleNeurons);
    MiddleLayer middle_layer2(&middle_layer1, kMiddleNeurons);
    OutputLayer output_layer (&middle_layer2, kOutputNeurons);

    for (std::size_t example = 0; example < kExamples; example++) {
        for (std::size_t neuron = 0; neuron < kInputNeurons; neuron++) {
            input_layer.SetValue(example, neuron, (float)images_buffer[example * kInputNeurons + neuron]/256);
        }

        output_layer.SetExpectedValue(example, labels_buffer[example], 1.0f);
    }

    middle_layer1.SetNormalRand();
    middle_layer2.SetNormalRand();
    output_layer .SetNormalRand();

    for (std::size_t example = kExamples; example < 2 * kExamples; example++) {
        for (std::size_t neuron = 0; neuron < kInputNeurons; neuron++) {
            input_layer.SetValue(example - kExamples, neuron, (float)images_buffer[example * kInputNeurons + neuron]/256);
        }

        output_layer.SetExpectedValue(example, labels_buffer[example], 1.0f);
    }

    // middle_layer1.LoadParamsFromFile(middle_layer1_saveload);
    // middle_layer2.LoadParamsFromFile(middle_layer2_saveload);
    // output_layer .LoadParamsFromFile(       output_saveload);

    // output_layer.EvalRecursive();

    bool isSaving = false;
    const float kStep = 200;
    const std::size_t kIterations = 1'000'000;
    for (std::size_t i = 0; i < kIterations; i++) {
        output_layer.ResetGradsRecursive();
        output_layer.EvalRecursive();

        std::cout << "Iteration " << i << ": loss = " << output_layer.GetLoss() << "\n"; 

        output_layer.BackpropagateRecursive(kStep);

        if (i % 100 == 0 && isSaving) {
            std::cout << "Saving...\n";
            // middle_layer1.SaveParamsToFile(middle_layer1_saveload);
            // middle_layer2.SaveParamsToFile(middle_layer2_saveload);
            // output_layer .SaveParamsToFile(       output_saveload);
        }

        // for (std::size_t j = 0; j < 10; j++) {
        //     std::cout << j << ": " << output_layer.GetExpectedValue(0, j) << " vs " << output_layer.GetNormOutput(0, j) << "\n";
        // }
    }
    output_layer.Dump();
}


#include "mnist/mnist.h"
#include "mnist/mnist_parser/file_to_buffer/file_to_buffer.h"
void TestMnistLib() {
    Mnist mnist("mnist/mnist_training_data/train-images.idx3-ubyte",
                "mnist/mnist_training_data/train-labels.idx1-ubyte",
                "mnist/mnist_weights",
                1,
                32);
    const float       kStep       = 20 * 1e-5f;
    const std::size_t nIterations = 20000;

    mnist.LoadWeights();
    std::cout << "Loading weights..." << std::endl;

    for (std::size_t iter = 0; iter < nIterations; iter++) {
        std::cout << "Iteration " << iter << std::endl;
        float loss = mnist.Eval();
        std::cout << "loss: " << loss << std::endl;
        mnist.Backpropagate(kStep);

        if (iter % 10 == 0 && iter != 0) {
            std::cout << "Saving weights" << std::endl;
            mnist.SaveWeights();
        }
    }

    return;

    FILE* drawing_data = fopen("mnist/drawing.bin", "rb");
    if (drawing_data == nullptr) {
        std::cerr << "First, draw using draw.py" << std::endl;
        return;
    }

    std::size_t buffer_size = 0;
    char* buffer = (char*)ftbPutFileToBuffer(&buffer_size, drawing_data);
    if (buffer == nullptr) {
        std::cerr << "Error putting to buffer" << std::endl;
        fclose(drawing_data);
        return;
    }

    float* float_buffer = reinterpret_cast<float*>(buffer);

    mnist.EvalImage(float_buffer);

    free(buffer);
}


void TestWriting() {
    FILE* drawing_data = fopen("drawing.bin", "rb");
    if (drawing_data == nullptr) {
        std::cerr << "First, draw using draw.py" << std::endl;
        return;
    }

    std::size_t buffer_size = 0;
    char* buffer = (char*)ftbPutFileToBuffer(&buffer_size, drawing_data);
    if (buffer == nullptr) {
        std::cerr << "Error putting to buffer" << std::endl;
        fclose(drawing_data);
        return;
    }


    free(buffer);
}
