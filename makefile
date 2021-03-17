CC=g++-8
NVCC=nvcc
EXECFOLDER=bin
OBJFOLDER=obj
PROJECTNAME=cuda-wah
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11 
#do cuda flags -arch=sm_10
CUDAFLAGS= -std=c++11 -c --compiler-bindir /usr/bin/g++-8 --output-file $(OBJFOLDER)/kernel.o
LIBS= -lpthread -lcudart 
LIBDIRS=-L/usr/local/cuda-10.1/lib64
INCDIRS=-I/usr/local/cuda-10.1/include
cuda-wah.o: kernel.cu
	$(NVCC) $(CUDAFLAGS) kernel.cu 
all: cuda-wah.o
	mkdir -p $(EXECFOLDER)
	mkdir -p $(OBJFOLDER)
	$(CC) -o $(EXECFOLDER)/$(PROJECTNAME) main.cpp $(OBJFOLDER)/kernel.o $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)
clean:
	rm -rf $(PROJECTNAME) *.o