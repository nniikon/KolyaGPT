#include "../include/smart_value.h"

#include <assert.h>
#include <cmath>
#include <iostream>


SmartValue::SmartValue(float value, OperationType oper, SmartValue* parent)
                            : value_(value), parent_oper_(oper), parent_(parent) {
    grad_ = 0.0f;
}


void SmartValue::SetValue(float value) {
    value_ = value;
}


float SmartValue::GetValue() const {
    return value_;
}


float SmartValue::GetGrad() const {
    return grad_;
}


void SmartValue::EvalGrad() {
    grad_ = 1.0f; // dx/dx is 1 by definition

    if (child1_) { child1_->EvalGradRecursive_(); }
    if (child2_) { child2_->EvalGradRecursive_(); }
}


void SmartValue::EvalGradRecursive_() {
    assert(parent_ != nullptr);

    float localSigm = 0.0f;
    switch(parent_oper_)
    {
        case OperationType::Sum:
            localSigm = 1.0f;
            break;

        case OperationType::Mul:
            localSigm = sibling_->value_;
            break;

        case OperationType::Sigm:
            localSigm = parent_->GetValue() * (parent_->GetValue() - 1);
            break;

        case OperationType::None:
        default:
            assert(0);
    }
    grad_ += localSigm * parent_->grad_;

    if (child1_) { child1_->EvalGradRecursive_(); }
    if (child2_) { child2_->EvalGradRecursive_(); }
}


void SmartValue::Dump() const {

    std::string   file_name = "graph.dot";
    std::string output_name = "smart_value_dump.png";

    std::ofstream out(file_name);
    // Exception??
    out << "digraph G {\n";
    out << "\tnode [shape=record];\n";

    DumpRecursive_(false, out);

    out << "}\n";
    out.close();

    std::string compile_cmd = "dot -Tpng " + file_name + " -o " + output_name;
    system(compile_cmd.c_str());
}


void SmartValue::DumpRecursive_(bool isSibling, std::ofstream& out) const {
    if (child1_) { child1_->DumpRecursive_(false, out); }
    if (child2_) { child2_->DumpRecursive_(true , out); } 

    out << "\tNode"   << this    << " [label=\"Value: " << value_ << " | Grad: " << grad_ << "\"];\n"; 
    if (parent_) {
        const char* op_str = nullptr;

        switch(parent_oper_) {
            case OperationType::Sum:    op_str = "+";       break;
            case OperationType::Mul:    op_str = "*";       break;
            case OperationType::Sigm:   op_str = "sigm";    break;

            case OperationType::None:
            default:
                assert(0);
                op_str = "NONE";
                break;
        }

        if (!isSibling) {
            out << "\top" << parent_ << " [label=\": " << op_str << "\"];\n"; 
        }

        out << "\tNode"   << this    << " -> " << "op" << parent_ << ";\n";
        out << "\top" << parent_ << " -> Node" <<         parent_ << ";\n";
    }
}


SmartValue* SmartValue::GetParent() const {
    return parent_;
}


void SmartValue::SetBinaryFamily(SmartValue* first, SmartValue* second, OperationType type) {
    first ->parent_ = this;
    second->parent_ = this;
    first ->parent_oper_ = type;
    second->parent_oper_ = type;
    first ->sibling_ = second;
    second->sibling_ = first;
    child1_ = first;
    child2_ = second;
}


void SmartValue::SetUnaryFamily(SmartValue* first, OperationType type) {
    first ->parent_ = this;
    first ->parent_oper_ = type;
    child1_ = first;
    child2_ = nullptr;
}


void SmartValue::Sum(SmartValue* first, SmartValue* second) {
    value_ = first->GetValue() + second->GetValue();
    SetBinaryFamily(first, second, OperationType::Sum);
}


void SmartValue::Mul(SmartValue* first, SmartValue* second) {
    value_ = first->GetValue() + second->GetValue();
    SetBinaryFamily(first, second, OperationType::Mul);
}


void SmartValue::Sigm(SmartValue* first) {
    value_ = 1 / (1 + expf(-first->GetValue()));
    SetUnaryFamily(first, OperationType::Sigm);
}
