#ifndef MLP_H_
#define MLP_H_

#include <cstddef>
#include "smart_matrix.h"

class Layer {
    public:
        Layer(std::size_t rows, std::size_t cols);
        virtual SmartMatrix* GetOutput() = 0;
        std::size_t          GetOutputRows() const;
        std::size_t          GetOutputCols() const;

    protected:
        SmartMatrix output_;
};

class InputLayer : public Layer {
    public:
        InputLayer(std::size_t n_inputs, std::size_t n_examples = 1);

        void SetValue(std::size_t i, std::size_t j, float value);

        void ResetGrads();

        SmartMatrix* GetOutput() override;

        std::size_t GetCols() const;
        std::size_t GetRows() const;

    private:
        const std::size_t n_inputs_;
        const std::size_t n_examples_;
};

class MiddleLayer : public Layer {
    public:
        MiddleLayer(Layer* input_layer, std::size_t n_outputs);

        void SetNormalRand();
        void Eval();
        void Backpropagate(float step);
        void ResetGrads();

        SmartMatrix* GetOutput() override;

    protected:
        Layer* const input_layer_;
        const std::size_t n_input_rows_;
        const std::size_t n_input_cols_;
        const std::size_t n_output_cols_;

        SmartMatrix weights_;
        SmartMatrix biases_;
        SmartMatrix unbiased_output_;
};

class OutputLayer : public MiddleLayer {

    public:
        OutputLayer(Layer* input_layer, std::size_t n_outputs);

        float EvalLoss();
        void Dump();
        void ResetGrads();
        float GetNormOutput(std::size_t example, std::size_t output);

        void SetExpectedValue(std::size_t example, std::size_t output, float value);

    private:
        SmartMatrix loss_;
        SmartMatrix expected_output_;
        SmartMatrix norm_output_;
};

#endif
