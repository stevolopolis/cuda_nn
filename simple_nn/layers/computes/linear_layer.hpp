#pragma once
 
#include "../nn_layer.hpp"

class LinearLayer : public NNLayer {
private:
    Matrix W, b;
    Matrix A, dA, Z;

    void initializeBiasWithZeros();
    void initializeWeightsWithRandomValues();

    void computeAndStoreBackprop(Matrix& dZ);
    void computeAndStoreOutput(Matrix& A); 
    void updateWeights(Matrix& dZ, float lr);
    void updateBias(Matrix& dZ, float lr);

public:
    LinearLayer(std::string name, Shape w_shape);
    ~LinearLayer();

    Matrix& forward(Matrix& A);
    Matrix& backward(Matrix& dZ, float lr);

    Matrix getWeightMatrix();
    Matrix getBiasVector();
};