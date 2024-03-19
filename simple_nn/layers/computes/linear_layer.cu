#include "linear_layer.hpp"

#include <random>
#include <assert.h>

/**
    * Kernel functions
*/
__global__ void linearLayerForward(float* A, float* W, float* b, float* Z,
                        int w_x_dim, int w_y_dim, int a_x_dim, int a_y_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Naive matmul
    if (row < w_y_dim && col < a_x_dim) {
        float sum = 0;
        for (int i = 0; i < w_x_dim; i++) {
            sum += W[row * w_x_dim + i] * A[i * a_x_dim + col];
        }
        Z[row * a_x_dim + col] = sum + b[row];
    }
}


__global__ void linearLayerBackward(float* W, float* dZ, float *dA,
    int w_x_dim, int w_y_dim, int dz_x_dim, int dz_y_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Naive matmul
    if (row < w_y_dim && col < dz_x_dim) {
        float sum = 0;
        for (int i = 0; i < w_x_dim; i++) {
            sum += W[i * w_y_dim + row] * dZ[i * dz_x_dim + col];
        }
        dA[row * dz_x_dim + col] = sum;
    }
}


__global__ void linearLayerUpdateWeights(float *W, float *dZ, float *A, float lr, int dz_x_dim, int dz_y_dim, int a_x_dim, int a_y_dim) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int dw_val = 0;
    if (idx_x < a_y_dim && idx_y < dz_y_dim) {
        for (int i = 0; i < a_x_dim; i++) {
            dw_val += dZ[idx_y * dz_x_dim + i] * A[idx_x * a_x_dim + i];
        }
        W[idx_y * a_y_dim + idx_x] -= lr * dw_val / a_x_dim;
    }
}

__global__ void linearLayerUpdateBias(float *b, float* dZ, float lr, int dz_x_dim, int dz_y_dim, int b_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dz_x_dim * dz_y_dim) {
        int col = idx % dz_x_dim;
        int row = idx / dz_x_dim;
        atomicAdd(&(b[row]), - lr * (dZ[row * dz_x_dim + col] / dz_x_dim));
    }
}


LinearLayer::LinearLayer(std::string name, Shape w_shape) :
    W(w_shape), b(w_shape.y, 1) 
{
    this->name = name;
    b.allocateMem();
    W.allocateMem();
    initializeWeightsWithRandomValues();
    initializeBiasWithZeros();
}


LinearLayer::~LinearLayer() {}


void LinearLayer::initializeBiasWithZeros() {
    for (int i = 0; i < b.shape.x; i++) {
        b[i] = 0;
    }

    b.copyHostToDevice();
}


void LinearLayer::initializeWeightsWithRandomValues() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < W.shape.x; i++) {
        for (int j = 0; j < W.shape.y; j++) {
            W[j*W.shape.x + i] = distribution(generator);
        }
    }

    W.copyHostToDevice();
}


Matrix& LinearLayer::forward(Matrix& A) {
    assert(W.shape.x == A.shape.y);

    this->A = A;
    Shape Z_shape(A.shape.x, W.shape.y);
    Z.allocateMemIfNotAllocated(Z_shape);

    computeAndStoreOutput(A);

    return Z;
}


Matrix& LinearLayer::backward(Matrix& dZ, float lr) {
    dA.allocateMemIfNotAllocated(A.shape);

    computeAndStoreBackprop(dZ);
    updateBias(dZ, lr);
    updateWeights(dZ, lr);

    return dA;
}


void LinearLayer::computeAndStoreBackprop(Matrix& dZ) {
    dim3 block_size(8,8);
    dim3 num_of_blocks((dA.shape.x + (block_size.x - 1)) / block_size.x,
                        (dA.shape.y + (block_size.y - 1)) / block_size.y);
    
    linearLayerBackward<<<num_of_blocks, block_size>>>(W.data_device.get(),
                                                    dZ.data_device.get(),
                                                    dA.data_device.get(),
                                                    W.shape.x, W.shape.y, dZ.shape.x, dZ.shape.y);
}



void LinearLayer::computeAndStoreOutput(Matrix& A) {
    dim3 block_size(8,8);
    dim3 num_of_blocks((Z.shape.x + (block_size.x - 1)) / block_size.x,
                        (Z.shape.y + (block_size.y - 1)) / block_size.y);
    
    linearLayerForward<<<num_of_blocks, block_size>>>(A.data_device.get(),
                                                    W.data_device.get(),
                                                    b.data_device.get(),
                                                    Z.data_device.get(),
                                                    W.shape.x, W.shape.y, A.shape.x, A.shape.y);
}


void LinearLayer::updateWeights(Matrix& dZ, float lr) {
    dim3 block_size(8,8);
    dim3 num_of_blocks((W.shape.x + (block_size.x - 1)) / block_size.x,
                        (W.shape.y + (block_size.y - 1)) / block_size.y);

    linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(W.data_device.get(),
                                                            dZ.data_device.get(),
                                                            A.data_device.get(),
                                                            lr,
                                                            dZ.shape.x,
                                                            dZ.shape.y,
                                                            A.shape.x,
                                                            A.shape.y);
}


void LinearLayer::updateBias(Matrix& dZ, float lr) {  
    dim3 block_size(64);
    dim3 num_of_blocks(dZ.shape.x * dZ.shape.y + (block_size.x - 1) / block_size.x);
    
    linearLayerUpdateBias<<<num_of_blocks, block_size>>>(b.data_device.get(),
                                                        dZ.data_device.get(),
                                                        lr,
                                                        dZ.shape.x,
                                                        dZ.shape.y,
                                                        b.shape.x);
}


Matrix LinearLayer::getWeightMatrix() {
    return W;
}


Matrix LinearLayer::getBiasVector() {
    return b;
}