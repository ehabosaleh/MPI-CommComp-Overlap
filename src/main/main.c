#include "nmpm.h"

int main(int argc, char *argv[]){
    int dim = DIM; 
    int compToPureCommRatio=COMP_TO_COMM_RATIO;

    for(int i=0;i<argc;i++){
        if(strncmp(argv[i],"--dim=",6)==0 ){
            dim=atoi(argv[i]+6);
            if(dim!=2 && dim!=3&& dim!=1){
                if(i==0){
                    fprintf(stderr, "Invalid dimension specified. Use 1 for 1D grid, 2 for 2D grid or 3 for 3D grid.\n");
                }
                return -1;
            }
        }
        else if(strncmp(argv[i],"--ratio=",8)==0){
            compToPureCommRatio=atoi(argv[i]+8);
            if(compToPureCommRatio<0){
                if(i==0){
                    fprintf(stderr, "Invalid CompToPureCommRatio specified. Use a non-negative integer.\n");
                }
                return -1;
            }
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

