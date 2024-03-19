CC=/usr/local/cuda/bin/nvcc

TARGETS = main

CAPABILITY=-arch=sm_86 # NVIDIA RTX A4000 (Ampere), CUDA 11.1+

all: $(TARGETS)

main: main.o neural_network.o bce.o linear_layer.o relu.o sigmoid.o matrix.o shape.o
	$(CC) $(CAPABILITY) -o main main.o neural_network.o bce.o linear_layer.o relu.o sigmoid.o matrix.o shape.o

main.o: main.cu data/data_coord.hpp neural_network.hpp layers/computes/linear_layer.hpp layers/activations/relu.hpp layers/activations/sigmoid.hpp losses/bce.hpp
	$(CC) $(CAPABILITY) -c main.cu

neural_network.o: neural_network.cu neural_network.hpp
	$(CC) $(CAPABILITY) -c neural_network.cu

bce.o: losses/bce.cu losses/bce.hpp
	$(CC) $(CAPABILITY) -c losses/bce.cu

linear_layer.o: layers/computes/linear_layer.cu layers/nn_layer.hpp
	$(CC) $(CAPABILITY) -c layers/computes/linear_layer.cu

relu.o: layers/activations/relu.cu layers/nn_layer.hpp
	$(CC) $(CAPABILITY) -c layers/activations/relu.cu

sigmoid.o: layers/activations/sigmoid.cu layers/nn_layer.hpp
	$(CC) $(CAPABILITY) -c layers/activations/sigmoid.cu

matrix.o: utils/matrix.cu utils/shape.hpp
	$(CC) $(CAPABILITY) -c utils/matrix.cu

shape.o: utils/shape.cu
	$(CC) $(CAPABILITY) -c utils/shape.cu

clean:
	rm -f *.o *.out