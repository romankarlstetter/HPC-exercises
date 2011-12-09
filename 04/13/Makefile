all: poisson_ian poisson_standard poisson_test

run: 
	./poisson_ian

poisson_ian: poisson_ian.cpp Makefile
	icc -O3 -xHost poisson_ian.cpp -o poisson_ian
	
poisson_standard: poisson_original.cpp
	icc -O3 -xHost poisson_original.cpp -o poisson_standard


poisson_test: poisson_test.cpp
	icc poisson_test.cpp -o poisson_test