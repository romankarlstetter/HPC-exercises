run:
	./quicksort_standard
	CILK_NWORKERS=2 ./quicksort_cilk
	CILK_NWORKERS=4 ./quicksort_cilk
	OMP_NUM_THREADS=2 ./quicksort_omp
	OMP_NUM_THREADS=4 ./quicksort_omp
all: quicksort_standard quicksort_cilk quicksort_omp

quicksort_standard: quicksort_original.c timer.o
	icc quicksort_original.c -o quicksort_standard timer.o

quicksort_cilk: quicksort_cilk.c timer.o
	icc quicksort_cilk.c -o quicksort_cilk timer.o

quicksort_omp: quicksort_openmp.c timer.o
	icc -openmp quicksort_openmp.c -o quicksort_omp timer.o
	
timer.o: timer.c timer.h
	icc timer.c -c