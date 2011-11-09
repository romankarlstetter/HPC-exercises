#!/bin/bash

#call me like this: 
#  salloc --ntasks=1 --cpus-per-task=8 --partition=ice1_inter ./test_problem_sizes


#load profile
. /etc/profile

#we wanna use openmp 3.1, lets load intel 12.1 compiler
module unload ccomp
module load ccomp/intel/12.1

icc timer.c quicksort.c -o quicksort -openmp

export OMP_NUM_THREADS=8

for i in 1000 10000 100000
do
for k in {1..50}
do
let j=$i*$k*2

./quicksort $j 132
done
done
