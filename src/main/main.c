#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "nmpm.h"

int main(int argc, char *argv[]){
    int dim = 2; 
    int compToPureCommRatio=100;
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"--dim")==0 && i+1<argc){
            dim=atoi(argv[i+1]);
        }
        else if(strcmp(argv[i],"--ratio")==0 && i+1<argc){
            compToPureCommRatio=atoi(argv[i+1]);
        }
        else if(strcmp(argv[i],"--help")==0){
            usage(argv[0]);
            return 0;
        }
    }
	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    run_overlap_benchmark(rank,size,dim, compToPureCommRatio);

    MPI_Finalize();
    return 0;
}

