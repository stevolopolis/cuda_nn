#pragma once

#include "../utils/matrix.hpp"

class Data {
protected:
    int batch_size, n_batches;
    std::vector<Matrix> X, Y;
public:
    std::vector<Matrix> get_data() {
        return X;
    }
    std::vector<Matrix> get_targets() {
        return Y;
    }
    Matrix get_single_data(int index) {
        return X[index];
    }
    Matrix get_single_target(int index) {
        return Y[index];
    }
    int get_num_batches() {
        return n_batches;
    }
    int get_batch_size() {
        return batch_size;
    }
};