#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include "layers/activations/relu.hpp"
#include "layers/activations/sigmoid.hpp"
#include "layers/computes/linear_layer.hpp"
#include "losses/bce.hpp"
#include "neural_network.hpp"
#include "data/data_coord.hpp"

#define EPOCH 100

int main() {
    CoordinateData dataset(1000, 50, 2);
    BCELoss criterion;

    NeuralNetwork nn(0.01);
    nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    nn.addLayer(new ReluActivation("relu_1"));
    // nn.addLayer(new LinearLayer("linear_2", Shape(30, 30)));
    // nn.addLayer(new ReluActivation("relu_2"));
    // nn.addLayer(new LinearLayer("linear_3", Shape(30, 30)));
    // nn.addLayer(new ReluActivation("relu_3"));
    nn.addLayer(new LinearLayer("linear_4", Shape(30, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid"));

    Matrix Y, targets;
    Matrix dY, error;
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        float loss = 0.0;
        float y_sum = 0;
        float target_sum = 0;
        for (int batch = 0; batch < dataset.get_num_batches() - 1; batch++) {
            Y = nn.forward(dataset.get_data().at(batch));
            targets = dataset.get_targets().at(batch);
            loss += criterion.cost(Y, targets);

            dY.allocateMemIfNotAllocated(Y.shape);
            error = criterion.dCost(Y, targets, dY);
        
            nn.backward(error);
            for (int k = 0; k < dataset.get_batch_size(); k++) {
                y_sum += Y[k];
                target_sum += targets[k];
            }
        }
        std::cout << "Y sum: " << y_sum << "\t Target sum: " << target_sum << std::endl;
        std::cout << "Epoch: " << epoch << "\t Loss: " << loss / (dataset.get_num_batches()-1) << std::endl;
    }

    // compute accuracy
    Y = nn.forward(dataset.get_data().at(dataset.get_num_batches() - 1));
    Y.copyDeviceToHost();
    targets = dataset.get_targets().at(dataset.get_num_batches() - 1);
    int n_correct = 0;
    for (int i = 0; i < dataset.get_batch_size(); i++) {
        int pred = Y[i] > 0 ? 1 : 0;
        if (pred == targets[i]) {
            n_correct++;
        }
    }
    float acc = static_cast<float>(n_correct) / static_cast<float>(dataset.get_batch_size());
    std::cout << "Accuracy: " << acc << std::endl;

    return 0;
}