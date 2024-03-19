#pragma once

#include "../utils/matrix.hpp"

class BCELoss {
public:
    float cost(Matrix preds, Matrix targets);
    Matrix dCost(Matrix preds, Matrix targets, Matrix dY);
};