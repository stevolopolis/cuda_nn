#include "linear_layer.hpp"

#include <random>

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


void LinearLayer::computeAndStoreBackprop(Matrix& dZ);
void LinearLayer::computeAndStoreOutput(Matrix& A); 
void LinearLayer::updateWeights(Matrix& dZ, float lr);
void LinearLayer::updateBias(Matrix& dZ, float lr);