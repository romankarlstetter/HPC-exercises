all: poisson_opencl_gcc poisson_opencl

poisson_opencl: poisson_opencl.cpp
	icc poisson_opencl.cpp -o poisson_opencl -lOpenCL

poisson_opencl_gcc: poisson_opencl.cpp
	gcc poisson_opencl.cpp -o poisson_opencl_gcc -lOpenCL

	
clean:
	rm dgemm dgemm_opencl dgemm_openmp opencl_info