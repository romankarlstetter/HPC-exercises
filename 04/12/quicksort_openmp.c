/** 
 * Quicksort implementation for practical course
 **/

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int threshold = 128;

void print_list(double *data, int length){
	int i;
	for(i = 0; i < length; i++)
		printf("%e\t", data[i]);
	printf("\n");
}

void quicksort(double *data, int length){
	if (length <= 1) return;

	//print_list(data, length);

	double pivot = data[0];
	double temp;
	int left = 1;
	int right = length - 1;

	do {
		while(left < (length - 1) && data[left] <= pivot) left++;
		while(right > 0 && data[right] >= pivot) right--;
		
		/* swap elements */
		if(left < right){
			temp = data[left];
			data[left] = data[right];
			data[right] = temp;
		}

	} while(left < right);
	
	if(data[right] < pivot){
		data[0] = data[right];
		data[right] = pivot;
	}

	//print_list(data, length);

	/* recursion */

        // we don't have to specify any data scope clauses because data is by default shared and everything else is private to the respective instances of the functions
        #pragma omp task final(right<threshold), shared(data)
        {
            quicksort(data, right);
        }

        #pragma omp task final(length - left<threshold), shared(data)
        {
            quicksort(&(data[left]), length - left);
        }

        //lets wait for both of our tasks, otherwise we get a segfault
        #pragma omp taskwait
}

int check(double *data, int length){
	int i;
	for(i = 1; i < length; i++)
		if(data[i] < data[i-1]) return 1;
	return 0;
}

int main(int argc, char **argv)
{
	int length;
	double *data;

	int mem_size;

	int i, j, k;

	length = 10000000;
        if(argc > 1){
                length = atoi(argv[1]);
        }
        if(argc > 2){
            threshold = atoi(argv[2]);
        }

	data = (double*)malloc(length * sizeof(double));
	if(0 == data){
		printf("memory allocation failed");
		return 0;
	}

	/* initialisation */
	srand(0);
	for (i = 0; i < length; i++){
		data[i] = (double)rand() / (double)RAND_MAX;
	}

	time_marker_t time = get_time();
	
	//print_list(data, length);

        #pragma omp parallel
        {
            #pragma omp single
            {
                quicksort(data, length);
            }
        }
        printf("Size of dataset; %d; elapsed time[s]; %e; threshold; %d \n", length, get_ToD_diff_time(time), threshold);


        //print_list(data, length);
	if(check(data, length) != 0)
                printf("ERROR\n");


	return(0);
}

