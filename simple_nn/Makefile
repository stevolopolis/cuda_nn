CC=/usr/local/cuda/bin/nvcc

TARGETS = matrix

CAPABILITY=-arch=sm_86 # NVIDIA RTX A4000 (Ampere), CUDA 11.1+

all: $(TARGETS)

relu.o: relu.cu nn_layer.hpp
	$(CC) $(CAPABILITY)  -c relu.cu

sigmoid.o: sigmoid.cu nn_layer.hpp
	$(CC) $(CAPABILITY)  -c sigmoid.cu

matrix.o: matrix.cu shape.hpp
	$(CC) $(CAPABILITY)  -c matrix.cu

shape.o: shape.cu
	$(CC) $(CAPABILITY)  -c shape.cu

clean:
	rm -f *.o *.out