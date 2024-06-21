#include "include/smart_value.h"
#include "include/smart_matrix.h"
#include "include/MLP.h"

#include <iostream>

void TestSmartValue ();
void TestSmartMatrix();
void TestMLP();

int main() {
    TestMLP();
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
    const std::size_t kExamples      = 2;
    const std::size_t kInputNeurons  = 4;
    const std::size_t kMiddleNeurons = 5;
    const std::size_t kOutputNeurons = 3;

    InputLayer   input_layer(kInputNeurons, kExamples);
    MiddleLayer middle_layer(&input_layer, kMiddleNeurons);
    OutputLayer output_layer(&middle_layer, kOutputNeurons);

    for (std::size_t i = 0; i < kExamples; i++) {
        for (std::size_t j = 0; j < kInputNeurons; j++) {
            input_layer.SetValue(i, j, (float)(i + j) / (kExamples + kInputNeurons));
        }

        for (std::size_t j = 0; j < kOutputNeurons; j++) {
            output_layer.SetExpectedValue(i, j, (float)(i+j) / (kExamples + kInputNeurons - 2));
        }
    }
    middle_layer.SetNormalRand();
    output_layer.SetNormalRand();

    for (std::size_t i = 0; i < 100'000; i++) {
        input_layer .ResetGrads();
        middle_layer.ResetGrads();
        output_layer.ResetGrads();

        middle_layer.Eval();
        float loss = output_layer.EvalLoss();

        std::cout << "Iteration " << i << ": loss = " << loss << "\n"; 

        for (size_t l = 0; l < kExamples; l++) {
            for (size_t j = 0; j < kOutputNeurons; j++) {
                std::cout << output_layer.GetNormOutput(l, j) << "\tExpected = " << (float)(l+j) / (kExamples + kInputNeurons - 2) << "\n";
            }
        }

        output_layer.Backpropagate(0.05f);
    }
    output_layer.Dump();
    
}
