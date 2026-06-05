#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "nom.h"

int main(int argc, char *argv[]){
	int rank, size;
    	MPI_Init(&argc, &argv);
    	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    	run_overlap_benchmark(rank,size);

    	MPI_Finalize();
    	return 0;
}

