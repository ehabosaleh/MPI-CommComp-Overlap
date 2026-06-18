#include "nmpm.h"

int main(int argc, char *argv[]){
    int dim = DIM;
    int dev=0;
    int compToPureCommRatio=COMP_TO_COMM_RATIO;
    
   

    for(int i=0;i<argc;i++){
        if(strncmp(argv[i],"--dim=",6)==0 ){
            dim=atoi(argv[i]+6);
            if(dim!=2 && dim!=3&& dim!=1){
                fprintf(stderr, "Invalid dimension specified. Use 1 for 1D grid, 2 for 2D grid or 3 for 3D grid.\n");
                return -1;
            }
        }
        else if(strncasecmp(argv[i],"--dev=",6)==0){
            dev=atoi(argv[i]+6);
            if(dev!=0&&dev!=1){
                fprintf(stderr, "Invalid device specified. Use 0 for CPU or 1 for GPU.\n");
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
    if(dev==1){
        #ifndef HAVE_CUDA
            fprintf(stderr,"CUDA is not installed \n");
            return -1;
        #endif
    }

	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(dev==0)
        run_overlap_benchmark(rank,size,dim,compToPureCommRatio);
    #if HAVE_CUDA
    else if(dev==1)
        run_overlap_benchmark_gpu(rank,size,dim,compToPureCommRatio);

    #endif
    
    MPI_Finalize();
    return 0;
}

