#include "include/smart_value.h"
#include "include/smart_matrix.h"

void TestSmartValue ();
void TestSmartMatrix();

int main() {
    TestSmartMatrix();
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
    SmartMatrix inputs (4, 3);
    SmartMatrix weights(3, 2);
    SmartMatrix outputs(4, 2);

    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 3; j++)
            inputs.SetValue(i, j, 1.0f);

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            weights.SetValue(i, j, 1.0f);

    outputs.Mul(&inputs, &weights);
    outputs.EvalGrad();

    outputs.Dump();
}
