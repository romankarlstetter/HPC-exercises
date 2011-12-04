#!/bin/sh

# go through output of /proc/kallsyms line by line (this is the default behaviour of awk) and 
# split the line by the default field separator (whitespace). then, look for all entries with T, D or R as the second field
# and create the corresponding entry which is writen to sysmpa.h

awk -- '{
	print $1" "$1/$2" "$1/$3" "$1/$4" "$1/$5" "$1/$6}
' testPerformance.gnuplot > testBandwidth.gnuplot
