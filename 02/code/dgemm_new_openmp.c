/** 
 * matrix matrix multiplication pattern for practical course
 **/

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int max(int a, int b){
	if(a >= b) return a;
	else return b;
}

int min(int a, int b){
	if(a <= b) return a;
	else return b;
}

int main(int argc, char **argv)
{
	int n, bs;
        double *a, *b, *c, *ref;

	int mem_size;

	int i, j, k, ii, jj, kk;

	/*char logfile_name[100];
	FILE *logfile_handle;*/

        n = 1000;
	bs = 32;
	if (argc > 1){
		n = atoi(argv[1]);
	}
	if (argc > 2){
		bs = atoi(argv[2]);
	}

	//sprintf(logfile_name, "logfile_dgemm.txt");
	//logfile_handle = freopen(logfile_name, "w", stdout);

	mem_size = n * n * sizeof(double);
	a = (double*)malloc(mem_size);
	b = (double*)malloc(mem_size);
        c = (double*)malloc(mem_size);
        ref = (double*)malloc(mem_size);
        if(0 == a || 0 == b || 0 == c){
		printf("memory allocation failed");
		return 0;
	}

	/* initialisation */
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			*(a + i * n + j) = (double)i + (double)j;
			*(b + i * n + j) = (double)(n - i) + (double)(n - j);
		}
	}
	memset(c, 0, mem_size);

        double flops;
        flops = 2.0 * n * n * n;
	time_marker_t time = get_time();

        #pragma omp parallel for private(j,k,ii,jj,kk)
	for(i = 0; i < n; i+= bs) {
		for(j = 0; j < n; j+=bs) {
			for(k = 0; k < n; k+=bs) {
				for(ii = i; ii < min(i+bs, n); ii++){
					for(jj = j; jj < min(j+bs, n); jj++){
                                                #pragma vector always
						for(kk = k; kk < min(k+bs, n); kk++) {
							c[ii * n + jj] += a[ii * n + kk] * b[kk * n + jj];
						}
					}
				} 
			}
		}
	}


        print_flops(flops, time);

        time = get_time();

	return(0);
}

