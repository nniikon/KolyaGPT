#ifndef SMART_VALUE_H_
#define SMART_VALUE_H_

#include <vector>
#include <fstream>

class SmartValue {
    public:

        enum class OperationType {
            None,
            Sum,
            Mul,
            Sigm,
        };

        SmartValue(float value = 0.0f, OperationType oper = OperationType::None,
                                                   SmartValue* parent = nullptr);
        void Sigm(SmartValue* first);
        void Sum (SmartValue* first, SmartValue* second);
        void Mul (SmartValue* first, SmartValue* second);
        void EvalGrad();
        void Dump() const;

        void        SetValue (float value);
        float       GetGrad  () const;
        float       GetValue () const;
        SmartValue* GetParent() const;

    private:
        float         value_;
        OperationType parent_oper_;
        float         grad_;
        SmartValue*   parent_;
        SmartValue*   sibling_;
        SmartValue*   child1_;
        SmartValue*   child2_;

        void DumpRecursive_(bool isSibling, std::ofstream& out) const;
        void EvalGradRecursive_();
        void SetBinaryFamily(SmartValue* first, SmartValue* second, OperationType type);
        void SetUnaryFamily (SmartValue* first,                     OperationType type);
};

#endif // SMART_VALUE_H_
