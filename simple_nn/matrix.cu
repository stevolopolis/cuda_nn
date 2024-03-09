#include "matrix.hpp"

Matrix::Matrix(size_t x_dim, size_t y_dim) :
    shape(x_dim, y_dim), 
    data_device(nullptr),
    data_host(nullptr),
    device_allocated(false),
    host_allocated(false) 
{}


Matrix::Matrix(Shape shape) :
    Matrix(shape.x, shape.y)
{}


void Matrix::allocateHostMem() {
    if (!host_allocated) {
        data_host = std::shared_ptr<float>(new float[shape.x * shape.y], [&](float* ptr) { delete[] ptr;});
        host_allocated = true;
    }
}


void Matrix::allocateDeviceMem() {
    if (!device_allocated) {
        float *device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        data_device = std::shared_ptr<float>(device_memory, [&](float *ptr) { cudaFree(ptr); });

        device_allocated = true;
    }
}


void Matrix::allocateMem() {
    allocateHostMem();
    allocateDeviceMem();
}


void Matrix::allocateMemIfNotAllocated(Shape shape) {
    if (!host_allocated && !device_allocated) {
        this->shape = shape;
        allocateMem();
    }
}


void Matrix::copyHostToDevice() {
    if (host_allocated && device_allocated) {
        cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
    }
}


void Matrix::copyDeviceToHost() {
    if (host_allocated && device_allocated) {
        cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
    }
}