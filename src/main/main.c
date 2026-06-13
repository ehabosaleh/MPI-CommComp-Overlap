#include "nmpm.h"

int main(int argc, char *argv[]){
    int dim = DIM; 
    int compToPureCommRatio=COMP_TO_COMM_RATIO;
    for(int i=0;i<argc;i++){
        if(strncmp(argv[i],"--dim=",6)==0 ){
            dim=atoi(argv[i]+6);
        }
        else if(strncmp(argv[i],"--ratio=",8)==0){
            compToPureCommRatio=atoi(argv[i]+8);
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

    run_overlap_benchmark(rank,size,dim,compToPureCommRatio);

    MPI_Finalize();
    return 0;
}

