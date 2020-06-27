ARCH     = -m64
GPU_ARCH = 
JAVA_DIR = /usr/lib/jvm/default-java
CUDA_INC = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

#############################################################################################

SRC_FILE1 = gpu.cu
SRC_FILE2 = gpu-concurrent.cu
COMPILER = nvcc --ptxas-options=-v $(GPU_ARCH) -Xcompiler -fPIC -I$(CUDA_INC) -L$(CUDA_LIB)

todo: gpu
gpu: gpu.cu
	$(JAVA_DIR)/bin/javac -classpath ../../../../target/CAIM-GPU-1.0-jar-with-dependencies.jar -h jni ../../java/weka/filters/supervised/attribute/CAIMGPU.java
	$(COMPILER) -O3 $(ARCH) -shared -I. -I$(JAVA_DIR)/include/linux -I$(JAVA_DIR)/include -c $(SRC_FILE1) -o gpu.o $(FLAGS) -lpthread -lm
	g++ $(ARCH) -O3 -fPIC -shared -I$(CUDA_INC) -L$(CUDA_LIB) -Wl,-soname,gpu gpu.o -o libgpu.so -lc -lpthread -lcuda -lcudart
clean:
	rm gpu.o libgpu.so