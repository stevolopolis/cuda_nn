#include "nn_layer.hpp"

class SigmoidActivation : public NNLayer {
private: 
    Matrix A, Z, dZ;
public:
    SigmoidActivation(std::string name);
    ~SigmoidActivation();

    Matrix& forward(Matrix& Z);
    Matrix& backward(Matrix& dA, float lr = 0.01);
};