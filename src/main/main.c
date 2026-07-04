#include "nmpm.h"

int main(int argc, char *argv[]){
    int dim = DIM;
    int dev=0;
    int compToPureCommRatio=COMP_TO_COMM_RATIO;
    long min_bytes=MIN_MESSAGE_SIZE;
    long max_bytes=MAX_MESSAGE_SIZE;
    int do_progress=0;
    int compute_bound=1;
    
    for(int i=0;i<argc;i++){
        if(strncmp(argv[i],"--dim=",6)==0 ){
            dim=atoi(argv[i]+6);
            if(dim!=2 && dim!=3&& dim!=1){
                fprintf(stderr, "Invalid dimension specified. Use 1 for 1D grid, 2 for 2D grid or 3 for 3D grid.\n");
                return -1;
            }
        }
        if(strncmp(argv[i],"--with-progress=",16)==0 ){
            do_progress=atoi(argv[i]+16);
            if(do_progress<0){
                fprintf(stderr, "Invalid input: 1 for enable progress; 0 for default\n");
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
        else if(strncasecmp(argv[i],"--max-bytes=",12)==0){
            max_bytes=atoi(argv[i]+12);
            if(max_bytes<=0){
                fprintf(stderr, "--max-bytes: Invalid message size.\n");
                return -1;
            }
        }

        else if(strncasecmp(argv[i],"--min-bytes=",12)==0){
            min_bytes=atoi(argv[i]+12);
            if(min_bytes<=0){
                fprintf(stderr, "--min-bytes: Invalid message size.\n");
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
        else if(strncmp(argv[i],"--compute-bound=",16)==0){
            compute_bound=atoi(argv[i]+16);
            if(compute_bound!=0 && compute_bound!=1){
                fprintf(stderr, "Invalid compute-bound flag specified. Use 0 for memory-bound or 1 for compute-bound.\n");
                return -1;
            }
        }
        else if(strncmp(argv[i],"--help",6)==0){
            usage(argv[0]);
            return 0;
        }
        else if(strncmp(argv[i],"--h",3)==0){
            usage(argv[0]);
            return 0;
        }
        else if(strncmp(argv[i],"-h",2)==0){
            usage(argv[0]);
            return 0;
        }
        else if(strncmp(argv[i],"-help",5)==0){
            usage(argv[0]);
            return 0;
        }

    }
    if(min_bytes>max_bytes){
        fprintf(stderr,"Maximum message size must be larger than minimum message size\n ");
        return -1;
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

    if(dev==0){
         run_overlap_benchmark(rank,size,dim,compToPureCommRatio,min_bytes,max_bytes,compute_bound);
    }

    else if(dev==1){
	#if HAVE_CUDA    
        run_overlap_benchmark_gpu(rank,size,dim,compToPureCommRatio,min_bytes,max_bytes,do_progress,compute_bound);

	#else
	fprintf(stderr, "GPU mode requested, but this binary was built without CUDA support.\n");
    	MPI_Finalize();
    	return -1;

	#endif
    }

    MPI_Finalize();
    return 0;
}
