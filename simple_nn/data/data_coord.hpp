#include <assert.h>
#include <stdlib.h>

#include "data.hpp"

class CoordinateData : public Data {
private:
    void intialize_data(int batch_size, int n_batches, int x_dims, int y_dims) {
        for (int i = 0; i < batch_size; i++) {
            X.push_back(Matrix(Shape(batch_size, x_dims)));
            Y.push_back(Matrix(Shape(batch_size, y_dims)));

            X[i].allocateMem();
            Y[i].allocateMem();
        }
    }
    void populate_with_random_data() {
        for (int i = 0; i < n_batches; i++) {
            for (int j = 0; j < batch_size; j++) {
                X[i][j] = static_cast<float>(rand()) / RAND_MAX - 0.5;
                X[i][j + batch_size] = static_cast<float>(rand()) / RAND_MAX - 0.5;

                if ((X[i][j] > 0 && X[i][j + batch_size] > 0) || (X[i][j] < 0 && X[i][j + batch_size] < 0)) {
                    Y[i][j] = 1;
                } else {
                    Y[i][j] = 0;
                }
            }
        }
        return;
    }
public:
    CoordinateData(int num_samples = 5000, int batch_size = 100, int x_dims = 2) {
        assert(num_samples % batch_size == 0);
        this->n_batches = num_samples / batch_size;
        this->batch_size = batch_size;
        intialize_data(batch_size, n_batches, x_dims, 1);
        populate_with_random_data();
    }
};