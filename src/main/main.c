#include "ccob.h"

int main(int argc, char *argv[]){
    int dim = DIM;
    int dev=0;
    int compToPureCommRatio=COMP_TO_COMM_RATIO;
    size_t min_bytes=MIN_MESSAGE_SIZE;
    size_t max_bytes=MAX_MESSAGE_SIZE;
    int do_progress=0;
    int enable_thread=0;
    int compute_bound=1;
    memory_mode_t memory_mode=MEMORY_MODE_TRIAD;
    const char *input_error_msg = NULL;
    
    for(int i=0;i<argc;i++){
        if(strncmp(argv[i],"--dim=",6)==0 ){
            dim=atoi(argv[i]+6);
            if(dim!=2 && dim!=3&& dim!=1){
                input_error_msg = "Invalid dimension specified. Use 1 for 1D grid, 2 for 2D grid or 3 for 3D grid.";
                break;
            }
        }
        if(strncmp(argv[i],"--with-progress=",16)==0 ){
            do_progress=atoi(argv[i]+16);
            if(do_progress<0){
                input_error_msg = "Invalid input: 1 for enable progress; 0 for default";
                break;
            }
        }
        else if(strncasecmp(argv[i],"--dev=",6)==0){
            char* dev_str = argv[i] + 6;
            if (strcasecmp(dev_str, "cpu") == 0) {
                dev = 0;
            } else if (strcasecmp(dev_str, "gpu") == 0) {
                dev = 1;
            } else {
                input_error_msg = "Invalid device specified. Use 'cpu' or 'gpu'.";
                break;
            }
        }
        else if(strncasecmp(argv[i],"--max-bytes=",12)==0){
            max_bytes=parse_size(argv[i]+12);
            if(max_bytes<=0){
                input_error_msg = "Invalid message size for --max-bytes.";
                break;
            }
        }

        else if(strncasecmp(argv[i],"--min-bytes=",12)==0){
            min_bytes=parse_size(argv[i]+12);
            if(min_bytes<=0){
                input_error_msg = "Invalid message size for --min-bytes.";
                break;
            }
        }
        else if(strncasecmp(argv[i],"--dev=",6)==0){
            dev=atoi(argv[i]+6);
            if(dev!=0&&dev!=1){
                input_error_msg = "Invalid device specified. Use 0 for CPU or 1 for GPU.";
                break;
            }
        }
        else if(strncmp(argv[i],"--ratio=",8)==0){
            compToPureCommRatio=atoi(argv[i]+8);
            if(compToPureCommRatio<0){
                if(i==0){
                    input_error_msg = "Invalid CompToPureCommRatio specified. Use a non-negative integer.";
                }
                break;
            }
        }
        else if(strncmp(argv[i],"--compute-bound=",16)==0){
            compute_bound=atoi(argv[i]+16);
            if(compute_bound!=0 && compute_bound!=1){
                input_error_msg = "Invalid compute-bound flag specified. Use 0 for memory-bound or 1 for compute-bound.";
                break;
            }
        }
        else if(strncmp(argv[i],"--memory-mode=",14)==0){
            const char* mode_str = argv[i] + 14;
            if (strcmp(mode_str, "triad") == 0) {
                memory_mode = MEMORY_MODE_TRIAD;
            } else if (strcmp(mode_str, "copy") == 0) {
                memory_mode = MEMORY_MODE_COPY;
            } else if (strcmp(mode_str, "scale") == 0) {
                memory_mode = MEMORY_MODE_SCALE;
            } else if (strcmp(mode_str, "add") == 0) {
                memory_mode = MEMORY_MODE_ADD;
            } else {
                input_error_msg = "Invalid memory mode specified. Use 'triad', 'copy', 'scale', or 'add'.";
                break;
            }
        }
        else if(strncmp(argv[i],"--progress-thread=",18)==0){
                enable_thread=atoi(argv[i]+18);
                if(enable_thread<0){
                    input_error_msg = "Invalid progress-thread flag specified. Use 0 for using thread for progressing or 1 for manual progress.";
                    break;
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
        input_error_msg = "Maximum message size must be larger than minimum message size";
    }
    if(dev==1){
        #ifndef HAVE_CUDA
            input_error_msg = "CUDA is not installed";
        #endif
    }

	int rank, size;
    if(do_progress){
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

        if(provided < MPI_THREAD_MULTIPLE) {
            input_error_msg = "MPI_THREAD_MULTIPLE not supported";
        }
    }
    else{
        MPI_Init(&argc, &argv);
    }
    
    MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        if(dim==2){
            if(sqrt(size)*sqrt(size)!=size){
                input_error_msg = "Number of processes must be a perfect square for 2D grid";
            }
        }
        if(dim==3){
            if(cbrt(size)*cbrt(size)*cbrt(size)!=size){
                input_error_msg = "Number of processes must be a perfect cube for 3D grid";
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(input_error_msg!=NULL){
        if(rank==0){
            fprintf(stderr,"Error: %s\n",input_error_msg);
            usage(argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    if(dev==0){
         run_overlap_benchmark(rank,size,dim,compToPureCommRatio,min_bytes,max_bytes,compute_bound,memory_mode,do_progress);
    }

    else if(dev==1){
	#if HAVE_CUDA    
        run_overlap_benchmark_gpu(rank,size,dim,compToPureCommRatio,min_bytes,max_bytes,do_progress,enable_thread,compute_bound,memory_mode);

	#else
	fprintf(stderr, "GPU mode requested, but this binary was built without CUDA support.\n");
    	MPI_Finalize();
    	return -1;

	#endif
    }

    MPI_Finalize();
    return 0;
}
