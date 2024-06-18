#include "include/smart_value.h"
#include <iostream>

int main() {
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
    std::cerr << "success";
}
