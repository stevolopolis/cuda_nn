#include <assert.h>
#include <math.h>

#include "../utils/matrix.hpp"
#include "bce.hpp"

__global__ void bce(float *preds, float *targets, int size, float *cost) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        float partial_cost = targets[idx] * logf(preds[idx]) + (1-targets[idx]) * logf(1-preds[idx]);
        atomicAdd(cost, - partial_cost / size);
    }
}


__global__ void dbce(float *preds, float *targets, float *dY, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        dY[idx] = - (targets[idx] / preds[idx] - (1 - targets[idx]) / (1 - preds[idx]));
    }
}


float BCELoss::cost(Matrix preds, Matrix targets) {
    assert(preds.shape.x == targets.shape.x);

    float *cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(64);
    dim3 num_of_blocks(preds.shape.x + block_size.x - 1 / block_size.x);
    bce<<<num_of_blocks, block_size>>>(preds.data_device.get(), targets.data_device.get(), preds.shape.x, cost);
    cudaDeviceSynchronize();

    float cost_val = *cost;
    cudaFree(cost);

    return cost_val;
}


Matrix BCELoss::dCost(Matrix preds, Matrix targets, Matrix dY) { 
    assert(preds.shape.x == targets.shape.x);

    dim3 block_size(64);
    dim3 num_of_blocks(preds.shape.x + block_size.x - 1 / block_size.x);
    dbce<<<num_of_blocks, block_size>>>(preds.data_device.get(), targets.data_device.get(), dY.data_device.get(), preds.shape.x);

    return dY;
}