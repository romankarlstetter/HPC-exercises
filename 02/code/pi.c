#include <stdio.h>
#include "timer.h"

#define N 50000000
#define h 1.0/N

double phi(double x)
{
    return 1.0/(1.0 + x * x);
}

double calculate_pi_openmp_critical()
{
    double sum = 0;
    double tmp_phi;

    // merge omp parallel and for pragmas, as we have only this one loop
    #pragma omp parallel for shared(sum), private(tmp_phi) schedule(static)
    for(int i = 0; i<N; i++)
    {
        tmp_phi = phi(h*i + h/2);
        #pragma omp critical
        {
            sum += tmp_phi;
        }
    }

    return sum*4*h;
}


double calculate_pi_openmp_reduction()
{
    double sum = 0;
    double tmp_phi;

    // merge omp parallel and for pragmas, as we have only this one loop
    #pragma omp parallel for reduction(+:sum), schedule(static), private(tmp_phi)
    for(int i = 0; i<N; i++)
    {
        tmp_phi = phi(h*i + h/2);
        sum += tmp_phi;
    }

    return sum*4*h;
}


double calculate_pi_sequential()
{
    double sum = 0;
    double tmp_phi;

    for(int i = 0; i<N; i++)
    {
        tmp_phi = phi(h*i + h/2);
        sum += tmp_phi;
    }
    return sum*4*h;
}


int main(int argc, char **argv)
{
    printf("\n");
    time_marker_t time;
    time = get_time();
    double pi = calculate_pi_sequential();
    printf("pi sequential       = %1.15f, time needed: %f secs, %f ticks\n", pi, get_ToD_diff_time(time), get_ticks_diff_time(time));

    time = get_time();
    pi = calculate_pi_openmp_critical();
    printf("pi openmp critical  = %1.15f, time needed: %f secs, %f ticks\n", pi, get_ToD_diff_time(time), get_ticks_diff_time(time));

    time = get_time();
    pi = calculate_pi_openmp_reduction();
    printf("pi openmp reduction = %1.15f, time needed: %f secs, %f ticks\n", pi, get_ToD_diff_time(time), get_ticks_diff_time(time));
}
