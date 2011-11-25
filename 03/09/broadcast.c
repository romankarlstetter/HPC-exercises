#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 13

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

int mpi_err, size, rank;
MPI_Status status;

double doubleArray[N];

void trivial(){
	int tag_trivial = 1232;
	if(rank == 0){
		MPI_Request *req = malloc(sizeof(MPI_Request)*(size-1));
		MPI_Status *status = malloc(sizeof(MPI_Status)*(size-1));
		for (int i = 1; i<size; i++){
			MPI_Isend(doubleArray, N, MPI_DOUBLE, i, tag_trivial, MPI_COMM_WORLD, &req[i-1]);
		}
		MPI_Waitall(size-1, req, status);
		free(req);
		free(status);
	} else {
		MPI_Request req;
		MPI_Status stat;
		MPI_Recv(doubleArray, N, MPI_DOUBLE, 0, tag_trivial, MPI_COMM_WORLD, &stat);
		//printf("Process %d received the array.\n ", rank);
	}
}

//handle processes from start to end (both including)
void tree_rec(int start, int end){
	MPI_Status stat;
	int tag = 234234;
	int indexMiddle = (start+end +1)/2;
	if (start == end) // end of recursion
		return;
	if(rank == start){ // only start has to send
		MPI_Send(doubleArray, N, MPI_DOUBLE, indexMiddle, tag, MPI_COMM_WORLD);
		printf("send from %d to %d\n", rank, indexMiddle);
	} if(rank == indexMiddle) { // receive
		MPI_Recv(doubleArray, N, MPI_DOUBLE, start, tag, MPI_COMM_WORLD, &stat);
		printf("recv in %d from %d\n", rank, start);
	}
	tree_rec(start, indexMiddle -1);
	tree_rec(indexMiddle, end);
}

void tree(){
	tree_rec(0, size-1);
}

void bonus(){
	int right_neighbour;
	int left_neighbour;
	if (rank == size - 1){
		right_neighbour = 1;
		left_neighbour = rank - 1;
	} else if (rank == 1) {
		right_neighbour = rank + 1;
		left_neighbour = size - 1;
	} else {
		right_neighbour = rank + 1;
		left_neighbour = rank - 1;
	}
	
	double elems_per_package = ((double)N) / ((double)size-1);
	int *elemsToSend = malloc(sizeof(int) * (size - 1));
	double **arrayStart = malloc(sizeof(double*) * (size - 1));
	for(int i = 0; i<size-1; i++){
		int ets = (int)(elems_per_package*(i+1)) - (int)(elems_per_package*(i));
		elemsToSend[i]= max(ets, 1); // send at least 1 elem to every proces
		arrayStart[i] = &doubleArray[(int)(elems_per_package*(i))];
	}
	
	int init_tag = 23423432;
	if(rank == 0){
		MPI_Request *req = malloc(sizeof(MPI_Request)*(size-1));
		MPI_Status *status = malloc(sizeof(MPI_Status)*(size-1));
		for(int i = 0; i<size-1; i++){
			MPI_Isend(arrayStart[i], elemsToSend[i], MPI_DOUBLE, i+1, init_tag, MPI_COMM_WORLD, &req[i]);
		}
		MPI_Waitall(size-1, req, status);
		free(req);
		free(status);
	} else {
		MPI_Status stat;
		MPI_Recv(arrayStart[rank-1], elemsToSend[rank-1], MPI_DOUBLE, 0, init_tag, MPI_COMM_WORLD, &stat);
	}
	
	if(rank != 0 ){ // all communication of process 0 is finished now
		for(int i= 0; i<size - 2; i++){
			MPI_Request req;
			MPI_Status stat;
			int indexRecv = (left_neighbour-i-1 + (size - 1))%(size-1);
			MPI_Irecv(arrayStart[indexRecv], elemsToSend[indexRecv], MPI_DOUBLE, left_neighbour, indexRecv, MPI_COMM_WORLD, &req);
			int indexSend = (rank - i - 1 + (size - 1))%(size-1);
			MPI_Send(arrayStart[indexSend], elemsToSend[indexSend], MPI_DOUBLE, right_neighbour, indexSend, MPI_COMM_WORLD);
			MPI_Wait(&req, &stat);
		}
	}
	
	free(elemsToSend);
	free(arrayStart);
}

void print_array_helperfun(){
	printf("Array of process %d:\n", rank);
	for(int i = 0; i<N; i++){
		printf("%d: %f\n", i, doubleArray[i]);
	}
}

//little function to print the arrays of all the processes with increasing process rank, so that we have a clean output
void print_array() {
	int dummy[1];
	int tokentag = rand();
	if(rank == 0){
		print_array_helperfun();
		MPI_Send(&dummy, 1, MPI_INT, 1, tokentag, MPI_COMM_WORLD);
	} else {
		MPI_Status stat;
		MPI_Recv(&dummy, 1, MPI_INT, rank-1, tokentag, MPI_COMM_WORLD, &stat);
		print_array_helperfun();
		if(rank != size-1)
			MPI_Send(&dummy, 1, MPI_INT, rank+1, tokentag, MPI_COMM_WORLD);
	}
}

void init_array(){
	// initialize array values
	if(rank == 0){
		for (int i = 0; i<N; i++){
			doubleArray[i] = i*11+0.1;//rand()/(double)RAND_MAX;
		}
	} else {
		for (int i = 0; i<N; i++){
			doubleArray[i] = 0; // reset to zero
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

main (int argc, char *argv[]){

	MPI_Init(&argc, &argv);
	if(mpi_err != MPI_SUCCESS){
		printf("Cannot initialize MPI!\n");
		MPI_Finalize();
		exit(0);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	init_array();
	
	trivial();
	MPI_Barrier(MPI_COMM_WORLD);
	
	init_array();
	
	tree();
	MPI_Barrier(MPI_COMM_WORLD);

	
	print_array();
	
	init_array();
	bonus();

	MPI_Finalize();
}

