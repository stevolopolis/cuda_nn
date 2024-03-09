CC=/usr/local/cuda/bin/nvcc

TARGETS = matrix

CAPABILITY=-arch=sm_86 # NVIDIA RTX A4000 (Ampere), CUDA 11.1+

all: $(TARGETS)

matrix: matrix.o shape.o
	$(CC) $(CAPABILITY) -o matrix.out matrix.o shape.o

sigmoid: sigmoid.o matrix.o
	$(CC) $(CAPABILITY) -o sigmoid.out sigmoid.o matrix.o

sigmoid.o: sigmoid.cu nn_layer.hpp
	$(CC) $(CAPABILITY)  -c sigmoid.cu

matrix.o: matrix.cu shape.hpp
	$(CC) $(CAPABILITY)  -c matrix.cu

shape.o: shape.cu
	$(CC) $(CAPABILITY)  -c shape.cu

clean:
	rm -f *.o *.out