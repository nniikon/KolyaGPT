#include "include/smart_value.h"
#include "include/smart_matrix.h"
#include "include/MLP.h"
#include "mnist/mnist_parser/mnist_parser.h"

#include <iostream>
#include <assert.h>
#include <immintrin.h>

void TestSmartValue ();
void TestSmartMatrix();
void TestMLP();
void TestMnistParser();

void TrainMnist();

int main() {
    TrainMnist();
}


void TestSmartValue() {
    SmartValue x1(0.1f);
    SmartValue x2(0.2f);
    SmartValue w1(0.2f);
    SmartValue w2(0.4f);
    SmartValue b (0.5f);
    SmartValue x1w1;
    SmartValue x2w2;
    SmartValue x1w1_x2w2;
    SmartValue res;
    x1w1.Mul(&x1, &x1);
    x2w2.Mul(&x2, &w2);
    x1w1_x2w2.Sum(&x1w1, &x2w2);
    res.Sigm(&x1w1_x2w2);
    res.EvalGrad();

    res.Dump();
}

void TestSmartMatrix() {
    const std::size_t kExampleInputs = 1;
    const std::size_t kInputSize     = 3;
    const std::size_t kOutputSize    = 2;

    SmartMatrix input_layers   (kExampleInputs, kInputSize);
    SmartMatrix weights        (kInputSize, kOutputSize);
    SmartMatrix unbiased_output(kExampleInputs, kOutputSize);
    SmartMatrix     norm_output(kExampleInputs, kOutputSize);
    SmartMatrix expected_output(kExampleInputs, kOutputSize);
    SmartMatrix            loss(1, 1);

    for (std::size_t i = 0; i < kOutputSize; i++)
        expected_output.SetValue(0, i, (float)i / 4);

    input_layers.SetMatrixNormRand();
    weights     .SetMatrixNormRand();

    for (std::size_t i = 0; i < 1'000'000; i++) {
        weights.ResetGrad();
        input_layers.ResetGrad();
        unbiased_output.ResetGrad();
        norm_output.ResetGrad();
        loss.ResetGrad();

        unbiased_output.Mul(&input_layers, &weights);
        norm_output.Sigm   (&unbiased_output);
        loss.Loss          (&norm_output, &expected_output);

        loss.EvalGrad();

        //std::cout << "Iteration " << i << ": loss = " << loss.GetValue(0, 0) << "\n"; 
        weights.AdjustValues(0.01f);
    }

    for (size_t i = 0; i < kOutputSize; i++) {
        std::cout << "Elem " << i << " = " << norm_output.GetValue(0, i) << "\tExpected = " << expected_output.GetValue(0, i) << "\n";
    }

    loss.Dump();
}


void TestMLP() {
    const std::size_t kExamples      = 6000;
    const std::size_t kInputNeurons  = 28*28;
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
            output_layer.SetExpectedValue(i, j, (float)(i+j) / (kExamples + kInputNeurons - 2));
        }
    }
    middle_layer1.SetNormalRand();
    middle_layer2.SetNormalRand();
    output_layer.SetNormalRand();

    const float kStep = 0.000001f;
    const std::size_t kIterations = 10000;
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
    MnistParser mnist_parser("mnist/MNIST_data/train-images.idx3-ubyte",
                             "mnist/MNIST_data/train-labels.idx1-ubyte");

    MnistLabels mnist_labels = mnist_parser.GetMnistLabels();
    MnistImages mnist_images = mnist_parser.GetMnistImages();

    std::cout << "Images:\n\tn_images = " << mnist_images.n_images << "\n";
    std::cout << "\tgrid = " << mnist_images.n_cols << " x " << mnist_images.n_rows << "\n\n";

    std::cout << "Labels:\n\tn_labels = " << mnist_labels.n_labels<< "\n";
}


void TrainMnist() {
    MnistParser mnist_parser("mnist/MNIST_data/train-images.idx3-ubyte",
                             "mnist/MNIST_data/train-labels.idx1-ubyte");

    MnistImages mnist_images = mnist_parser.GetMnistImages();
    MnistLabels mnist_labels = mnist_parser.GetMnistLabels();

    uint8_t* images_buffer = mnist_images.buffer;
    uint8_t* labels_buffer = mnist_labels.buffer;

    assert(mnist_labels.n_labels == mnist_images.n_images);

    //const std::size_t kExamples      = static_cast<std::size_t>(mnist_labels.n_labels);
    const std::size_t kExamples      = 10000;
    const std::size_t kInputNeurons  = mnist_images.n_cols * mnist_images.n_rows;
    const std::size_t kMiddleNeurons = 12;
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

    //middle_layer1.SetNormalRand();
    //middle_layer2.SetNormalRand();
    //output_layer .SetNormalRand();

    const char* middle_layer1_saveload = "mnist_data/middle1.data";
    const char* middle_layer2_saveload = "mnist_data/middle2.data";
    const char*        output_saveload = "mnist_data/output.data";

    middle_layer1.LoadParamsFromFile(middle_layer1_saveload);
    middle_layer2.LoadParamsFromFile(middle_layer2_saveload);
    output_layer .LoadParamsFromFile(       output_saveload);

    const float kStep = 2 * 1e-4f;
    const std::size_t kIterations = 1'000'000;
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

        if (i % 10 == 0) {
            std::cout << "Saving...\n";
            middle_layer1.SaveParamsToFile(middle_layer1_saveload);
            middle_layer2.SaveParamsToFile(middle_layer2_saveload);
            output_layer .SaveParamsToFile(       output_saveload);
        }
    }

}
