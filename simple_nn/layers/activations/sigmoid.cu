#include "sigmoid.hpp"

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}   


__global__ void sigmoidActivationForward(float *Z, float *A, int Z_x_dim, int Z_y_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Z_x_dim * Z_y_dim) {
        A[i] = sigmoid(Z[i]);
    }
}


__global__ void sigmoidActivationBackward(float *dA, float *dZ, float *Z, int Z_x_dim, int Z_y_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Z_x_dim * Z_y_dim) {
        float sigmoidZ = sigmoid(Z[i]);
        dZ[i] = dA[i] * sigmoidZ * (1 - sigmoidZ);
    }
}


SigmoidActivation::SigmoidActivation(std::string name) :
    name(name)
{ }


SigmoidActivation::~SigmoidActivation() {}


Matrix& SigmoidActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.x * Z.shape.y + block_size.x - 1) / block_size.x);

    sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);

    return A;
}


Matrix& SigmoidActivation::backward(Matrix& dA, float lr) {
    dZ.allocateMemIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.x * Z.shape.y + block_size.x - 1) / block_size.x);

    sigmoidActivationBackward<<<num_of_blocks, block_size>>>(dA.data_device.get(),
                                                             dZ.data_device.get(),
                                                             Z.data_device.get(),
                                                             Z.shape.x,
                                                             Z.shape.y);

    return dZ;
}