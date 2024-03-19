#pragma once
 
#include <iostream>

#include "../utils/matrix.hpp"

class NNLayer {
protected:
    std::string name;

public:
    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix& A) = 0;
    virtual Matrix& backward(Matrix& dA, float lr) = 0;

    std::string getName() { return this->name; };
};

inline NNLayer::~NNLayer() { }