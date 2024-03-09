#include "shape.hpp"

#include <memory>

class Matrix {
private: 
    bool device_allocated;
    bool host_allocated;

    void allocateDeviceMem();
    void allocateHostMem();

public:
    Shape shape;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape shape);

    void allocateMem();
    void allocateMemIfNotAllocated(Shape shape);

    void copyHostToDevice();
    void copyDeviceToHost();   
};