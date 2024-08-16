#ifndef MLP_H_
#define MLP_H_

#include <cstddef>
#include "smart_matrix.h"

class Layer {
    public:
        Layer(std::size_t rows, std::size_t cols, Layer* input_layer);
        virtual ~Layer();

        Layer(const Layer& other);
        Layer& operator=(const Layer& other);
        Layer(Layer&& other);
        Layer& operator=(Layer&& other);

        virtual SmartMatrix* GetOutput() = 0;
        virtual void        ResetGrads() = 0;
        std::size_t          GetOutputRows() const;
        std::size_t          GetOutputCols() const;
        Layer*               GetInputLayer() const;

        void                 SetInputLayer(Layer* layer);

        virtual void EvalRecursive()                    = 0;
        virtual void ResetGradsRecursive()              = 0;
        virtual void BackpropagateRecursive(float step) = 0;

    protected:
        Layer* input_layer_;
        SmartMatrix output_;
};

class InputLayer : public Layer {
    public:
        InputLayer(std::size_t n_inputs, std::size_t n_examples = 1);
        ~InputLayer();

        InputLayer(const InputLayer& other);
        InputLayer& operator=(const InputLayer& other);
        InputLayer(InputLayer&& other);
        InputLayer& operator=(InputLayer&& other);

        void SetValue(std::size_t i, std::size_t j, float value);

        void ResetGrads() override;

        SmartMatrix* GetOutput() override;

        void EvalRecursive()                    override;
        void ResetGradsRecursive()              override;
        void BackpropagateRecursive(float step) override;

        std::size_t GetCols() const;
        std::size_t GetRows() const;

    private:
        const std::size_t n_inputs_;
        const std::size_t n_examples_;
};

class MiddleLayer : public Layer {
    public:
        MiddleLayer(Layer* input_layer, std::size_t n_outputs);
        ~MiddleLayer();

        MiddleLayer(const MiddleLayer& other);
        MiddleLayer& operator=(const MiddleLayer& other);
        MiddleLayer(MiddleLayer&& other);
        MiddleLayer& operator=(MiddleLayer&& other);

        void SetNormalRand();
        virtual void Eval();
        void Backpropagate(float step);
        void ResetGrads() override;
        void SaveParamsToFile  (const char* file_name);
        void LoadParamsFromFile(const char* file_name);

        void EvalRecursive()                    override;
        void ResetGradsRecursive()              override;
        void BackpropagateRecursive(float step) override;

        SmartMatrix* GetOutput() override;

    protected:
        const std::size_t n_input_rows_;
        const std::size_t n_input_cols_;
        const std::size_t n_output_cols_;

        SmartMatrix weights_;
        SmartMatrix biases_;
        SmartMatrix unbiased_output_;
        SmartMatrix norm_output_;
};

class OutputLayer : public MiddleLayer {

    public:
        OutputLayer(Layer* input_layer, std::size_t n_outputs);
        ~OutputLayer();

        OutputLayer(const OutputLayer& other);
        OutputLayer& operator=(const OutputLayer& other);
        OutputLayer(OutputLayer&& other);
        OutputLayer& operator=(OutputLayer&& other);

        float GetLoss() const;
        float EvalLoss();
        void Dump();
        void Eval() override;
        void ResetGrads() override;
        float GetNormOutput(std::size_t example, std::size_t output);

        void  SetExpectedValue(std::size_t example, std::size_t output, float value);
        float GetExpectedValue(std::size_t example, std::size_t output);

        void EvalRecursive()                    override;
        void ResetGradsRecursive()              override;
        void BackpropagateRecursive(float step) override;

    private:
        SmartMatrix loss_;
        SmartMatrix expected_output_;
};

#endif
