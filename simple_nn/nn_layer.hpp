#include "matrix.hpp"

class NNLayer {
private:
    std:string name;

public:
    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix& A) = 0;
    virtual Matrix& backward(Matrix& dA, float lr) = 0;

    std:string getName() { return this->name; };
};