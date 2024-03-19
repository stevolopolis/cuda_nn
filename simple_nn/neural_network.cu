#include "neural_network.hpp"


NeuralNetwork::NeuralNetwork(float learning_rate) :
    lr(learning_rate)
{}


NeuralNetwork::~NeuralNetwork() {
    for (auto layer : layers) {
        delete layer;
    }
}


void NeuralNetwork::addLayer(NNLayer *layer) {
    this->layers.push_back(layer);
}


Matrix NeuralNetwork::forward(Matrix inputs) {
    Matrix Z = inputs;
    for (auto layer : layers) {
        Z = layer->forward(Z);
    }

    return Z;
}


void NeuralNetwork::backward(Matrix error) {
    for (auto layer = this->layers.rbegin(); layer != this->layers.rend(); layer++) {
        error = (*layer)->backward(error, lr);
    }

    cudaDeviceSynchronize();
}


std::vector<NNLayer *> NeuralNetwork::getLayers() const {
    return layers;
}