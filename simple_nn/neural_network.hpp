#pragma once

#include <vector>

#include "layers/nn_layer.hpp"
#include "utils/matrix.hpp"
#include "losses/bce.hpp"

class NeuralNetwork {
private:
    std::vector<NNLayer *> layers;
    BCELoss loss;

    float lr;
public:
    NeuralNetwork(float lr = 0.01);
    ~NeuralNetwork();

    void addLayer(NNLayer *layer);
    Matrix forward(Matrix inputs);
    void backward(Matrix error);

    std::vector<NNLayer *> getLayers() const;
};