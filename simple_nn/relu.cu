#include "relu.hpp"

__device__ float relu(float x) {
    return x < 0 ? 0 : x;
}   


__global__ void reluActivationForward(float *Z, float *A, int Z_x_dim, int Z_y_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Z_x_dim * Z_y_dim) {
        A[i] = relu(Z[i]);
    }
}


__global__ void reluActivationBackward(float *dA, float *dZ, float Z, int Z_x_dim, int Z_y_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Z_x_dim * Z_y_dim) {
        dZ[i] = dA[i] * (Z[i] > 0 ? 1 : 0);
    }
}


Matrix& ReluActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.x * Z.shape.y + block_size.x - 1) / block_size.x);

    reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);

    return A;
}


Matrix& ReluActivation::backward(Matrix& dA, float lr) {
    dZ.allocateMemIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.x * Z.shape.y + block_size.x - 1) / block_size.x);

    reluActivationBackward<<<num_of_blocks, block_size>>>(dA.data_device.get(),
                                                        dZ.data_device.get(),
                                                        Z.data_device.get(),
                                                        Z.shape.x,
                                                        Z.shape.y);

    return dZ;
}