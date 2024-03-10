#include "nn_layer.hpp"

class ReluActivation : public NNLayer {
private: 
    Matrix A, Z, dZ;
public:
    ReluActivation(std::string name);
    ~ReluActivation();

    Matrix& forward(Matrix& Z);
    Matrix& backward(Matrix& dA, float lr = 0.01);
};