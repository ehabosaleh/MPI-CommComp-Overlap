#include"nmpm.h"

void usage(char *prog_name) {
	fprintf(stderr, "Usage: %s [--dim=N] [--ratio=P] [--dev=gpu/cpu] [--with-progress=y/n] [--min-bytes=N] [--max-bytes=N] [--memory-mode=MODE]\n", prog_name);
	fprintf(stderr, "--dim: 1 for 1D grid, 2 for 2D grid, 3 for 3D grid\n");
	fprintf(stderr, "--ratio: Desired computation to pure communication time ratio (e.g., 50 for 50%%)\n");
	fprintf(stderr,"--dev: 0 to run the benchmark on the CPU or 1 to run the benchmark on the GPU\n");
	fprintf(stderr,"--with-progress: 1 to apply MPI_Testall to pending communication requests or 0 to rely on internal MPI progress\n");
	fprintf(stderr,"--min-bytes: Desired minimum number of bytes\n");
	fprintf(stderr,"--max-bytes: Desired maximum number of bytes\n");
	fprintf(stderr,"--compute-bound: Flag to indicate compute-bound benchmark\n");
	fprintf(stderr,"--memory-mode: Mode for memory-bound operations (triad, copy, scale, add)\n");
	fprintf(stderr,"Example: mpirun --hostfile hostfile -np <num_processes> ./test --dim=2 --ratio=50 --dev=1 --with-progress=1 --min-bytes=1024 --max-bytes=1048576 --compute-bound=0 --memory-mode=triad\n");
}

static int get_local_rank(){
	MPI_Comm local_comm;
	int local_rank=0;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
	MPI_Comm_rank(local_comm,&local_rank);
	MPI_Comm_free(&local_comm);
	return local_rank;
}
static int coordinates(int*dims,int*coords, int rank,int size, int dim){ 
	if(dim==3){
		dims[0] = dims[1] = dims[2] = (int)cbrt((double)size);
		int cbrt_size = (int)cbrt((double)dims[0]*dims[1]*dims[2]);
		if (cbrt_size * cbrt_size * cbrt_size != size) {
				if (rank == 0) {
					fprintf(stderr, "Number of processes must be a perfect cube for 3D grid\n");
				}
				MPI_Finalize();
				return -1;
			}
		coords[0] = rank / (cbrt_size * cbrt_size);
		coords[1] = (rank % (cbrt_size * cbrt_size)) / cbrt_size;
		coords[2] = rank % cbrt_size;
	}
	else if(dim==2){
		dims[0]=dims[1]=(int)sqrt((double)size);
		int sqrt_size=(int)sqrt((double)dims[0]*dims[1]);
		if(sqrt_size * sqrt_size != size) {
				if (rank == 0) {
					fprintf(stderr, "Number of processes must be a perfect square for 2D grid\n");
				}
				MPI_Finalize();
				return -1;
			}
		coords[0] = rank / sqrt_size;
		coords[1] = rank % sqrt_size;
		coords[2] = 0;
	}
	else if(dim==1){
		dims[0] = size;
		coords[0] = rank;
		coords[1] = 0;
		coords[2] = 0;
	}
	return 0;

}
static void find_neighbors(int*left,int*right,int*front,int*back,int*bottom,int*top, int *dims, int*coords, int rank, int dim){
	if(dim==3){
		*left = (coords[2]==0) ? MPI_PROC_NULL : rank - 1;
		*right = (coords[2]==dims[2]-1) ? MPI_PROC_NULL : rank + 1;
		*front = (coords[1]==0) ? MPI_PROC_NULL : rank - dims[2];
		*back = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + dims[2];
		*bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[2]*dims[1];
		*top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[2]*dims[1];
	}
	else if(dim==2){
		*left = (coords[1]==0) ? MPI_PROC_NULL : rank - 1;
		*right = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + 1;
		*top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[1];
		*bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[1];
	}
	else if(dim==1){
		*left = (coords[0]==0) ? MPI_PROC_NULL : rank - 1;
		*right = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + 1;
	}

}
static void post_sendrecv(int left,int right, int front, int back, int bottom, int top, int dim,char**send_buffers,char**recv_buffers, MPI_Request *reqs, int *req_count,long local_N){
	if (dim==3) {
		if(left != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(right != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(front != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(back != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(top != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(bottom != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
    } else if(dim==2){ 

		if(left != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(right != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(top != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(bottom != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
    }
	else if(dim==1){
		if(left != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
		if(right != MPI_PROC_NULL) {
			MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
			MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[(*req_count)++]);
		}
	}

}

int run_overlap_benchmark(int rank, int size, int dim, int compToPureCommRatio, long min_bytes,long max_bytes, int compute_bound,memory_mode_t memory_mode){ 
	int iter;
    double t_pure_total=0.0, t_comp_total=0.0, t_ovrl_total=0.0;
    double overlap=0.0;
    double  t_pure_global=0.0, t_compute_global=0.0, t_ovrl_global=0.0;
	double t_comp=0.0,t_ovrl=0.0;

	int dims[3], coords[3];
	int num_neighbors;
	int left=MPI_PROC_NULL, right=MPI_PROC_NULL, front=MPI_PROC_NULL, back=MPI_PROC_NULL, bottom=MPI_PROC_NULL, top=MPI_PROC_NULL;

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
		
	if(dim==3){
		coordinates(dims,coords,rank,size,3);
		num_neighbors=6;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,3);
	}
	else if(dim==2){
		coordinates(dims,coords,rank,size,2);
		num_neighbors=4;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,2);
	}
	else if(dim==1){
		coordinates(dims,coords,rank,size,1);
		num_neighbors=2;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,1);
	}

	init_memory_bound_buffers(256UL*1024UL*1024UL);
    if (rank==0) {
		if (dim==3) {
			printf("\nRunning 3D benchmark on CPU with ranks grid %dx%dx%d\n", dims[0], dims[1], dims[2]);
		} else if(dim==2){
			printf("\nRunning 2D benchmark on CPU with ranks grid %dx%d\n", dims[0], dims[1]);
		} else if(dim==1){
			printf("\nRunning 1D benchmark on CPU with ranks grid %d\n", dims[0]);
		}
		printf("Compute-bound benchmark: %s\n", compute_bound ? "Yes" : "No");
		if (!compute_bound) {
			printf("Memory mode: %s\n", memory_mode == TRIAD ? "Triad" : memory_mode == COPY ? "Copy" : memory_mode == SCALE ? "Scale" : "Add");
		}
		printf("With manual progress: %s\n", do_progress ? "Yes" : "No");
		printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n","Size (Bytes)","Communication(us)","Computation(us)","Actual Ratio %","Requested Ratio %","Overall","Overlapping %");
    }

    MPI_Request *reqs=(MPI_Request*)malloc(2*num_neighbors*sizeof(MPI_Request));
    for (long local_N=min_bytes;local_N <= max_bytes; local_N *= 2) {

        char *send_buffers[6];
        char *recv_buffers[6];
        	
		for (int i = 0; i < 6; i++) {
        	send_buffers[i] = (char *)malloc(local_N);
        	recv_buffers[i] = (char *)malloc(local_N);
        	for (long j = 0; j < local_N; j++) {
            	send_buffers[i][j] = 'a' + i;
            	recv_buffers[i][j] = 'a' + i;
        	}
        }
        for (iter =0; iter< MAX_ITER; iter++){
            int req_count=0;
			MPI_Barrier(MPI_COMM_WORLD);
            double init_time=MPI_Wtime();
            post_sendrecv(left,right,front,back,bottom,top,dim,send_buffers,recv_buffers,reqs,&req_count,local_N);
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

            if(iter>=SKIP) {
                t_pure_total += MPI_Wtime()-init_time;
            }
            		
    	}
        t_pure_total = 1e6 * t_pure_total/(MAX_ITER-SKIP);
		MPI_Allreduce(&t_pure_total, &t_pure_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        for (iter = 0; iter < MAX_ITER; iter++) {
            int req_count = 0;
			MPI_Barrier(MPI_COMM_WORLD);
            double init_time = MPI_Wtime();

            post_sendrecv(left,right,front,back,bottom,top,dim,send_buffers,recv_buffers,reqs,&req_count,local_N);	
            double targetComputeTime = (compToPureCommRatio/100.0)*t_pure_global;
            double tcomp_start = MPI_Wtime();
            compute_on_host((targetComputeTime/1e6),compute_bound,memory_mode);
            t_comp = MPI_Wtime()-tcomp_start;
			
            MPI_Waitall(req_count,reqs,MPI_STATUSES_IGNORE);

            t_ovrl=MPI_Wtime()-init_time;

            if(iter>=SKIP){
                t_comp_total += t_comp;
                t_ovrl_total += t_ovrl;
            }
        }

        t_comp_total = (t_comp_total*1e6)/(MAX_ITER-SKIP);
        t_ovrl_total= (t_ovrl_total*1e6)/(MAX_ITER-SKIP);
		
		MPI_Allreduce(&t_comp_total, &t_compute_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&t_ovrl_total, &t_ovrl_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if(rank == 0) {
			double actualRatio=100.0*(t_compute_global/t_pure_global);
			overlap= 100.0 * fmax(0.0,fmin(1.0,(t_pure_global+t_compute_global-t_ovrl_global)/fmin(t_pure_global, t_compute_global)));
            printf("%-20ld%-20.3f%-20.3f%-20.3f%-20d%-20.3f%-20.3f\n",local_N, t_pure_global, t_compute_global,actualRatio,compToPureCommRatio, t_ovrl_global, overlap);
        }
		t_comp_total=0;
		t_ovrl_total=0;
		t_pure_total=0;
		t_pure_global=0;
		t_compute_global=0;
		t_ovrl_global=0;
		overlap=0;

        for (int i = 0; i < 6; i++) {
        	free(send_buffers[i]);
        	free(recv_buffers[i]);
        }
    }
    free(reqs);
	return 0;
}

#if HAVE_CUDA

int run_overlap_benchmark_gpu(int rank, int size, int dim, int compToPureCommRatio, long min_bytes, long max_bytes,int do_progress,int compute_bound, memory_mode_t memory_mode) {
	int iter;
	int gpu_inner_iters;
	
	gpu_memory_calibration_t mem_cal={0,0,0,0};
	size_t elems_per_pass = ELEMS_PER_PASS;
	size_t max_elems = VECTOR_DIM_MEM;
	size_t N = max_elems;
	double measured_unit_us=0.0;

    double t_pure_total=0.0, t_comp_total=0.0, t_ovrl_total=0.0;
    double overlap=0.0;
    double  t_pure_global=0.0, t_compute_global=0.0, t_ovrl_global=0.0;
	double t_comp=0.0,t_ovrl=0.0;

	int dims[3], coords[3];
	int num_neighbors;
	int left=MPI_PROC_NULL, right=MPI_PROC_NULL, front=MPI_PROC_NULL, back=MPI_PROC_NULL, bottom=MPI_PROC_NULL, top=MPI_PROC_NULL;

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	int local_rank=get_local_rank();
	int device_count=0;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	CHECK_CUDA_ERROR(cudaSetDevice(local_rank%device_count));
	cudaDeviceProp prop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop,local_rank%device_count));
	
	cudaStream_t stream;
	CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
	int grid=prop.multiProcessorCount*4;
	int block=TPB_256;
	
	init_vector(N);


	if(compute_bound==1){
		gpu_inner_iters=calibrate_inner_iter(d_a,stream,grid,block,VECTOR_DIM_COMP,1,&measured_unit_us);
	}
	else{

		mem_cal=calibrate_memory_bound_kernel(d_c,d_a,d_b,stream,grid,block,elems_per_pass,max_elems,50,memory_mode);
		if (rank == 0) {
    		printf("elems_per_pass = %zu\n", elems_per_pass);
    		printf("max_elems      = %zu\n", max_elems);
    		printf("max passes     = %zu\n", max_elems / elems_per_pass);
    		printf("per vector     = %.3f MB\n",(double)(max_elems * sizeof(float)) / (1024.0 * 1024.0));
    		printf("3 vectors      = %.3f MB\n",(double)(3.0 * max_elems * sizeof(float)) / (1024.0 * 1024.0));
    		fflush(stdout);
		}
	}	
	if(dim==3){
		coordinates(dims,coords,rank,size,3);
		num_neighbors=6;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,3);
	}
	else if(dim==2){
		coordinates(dims,coords,rank,size,2);
		num_neighbors=4;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,2);
	}
	else if(dim==1){
		coordinates(dims,coords,rank,size,1);
		num_neighbors=2;
		find_neighbors(&left,&right,&front,&back,&bottom,&top,dims,coords,rank,1);
	}

	

	if (rank==0) {
			if (dim==3) {
				printf("\nRunning 3D benchmark on GPU with ranks grid %dx%dx%d\n", dims[0], dims[1], dims[2]);
			} else if(dim==2){
				printf("\nRunning 2D benchmark on GPU with ranks grid %dx%d\n", dims[0], dims[1]);
			} else if(dim==1){
				printf("\nRunning 1D benchmark on GPU with ranks grid %d\n", dims[0]);
			}
			printf("With manual progress: %s\n", do_progress ? "Yes" : "No");
			printf("Compute-bound: %s\n", compute_bound ? "Yes" : "No");
			if(!compute_bound) {
				printf("Memory-bound mode: %s\n", memory_mode == TRIAD ? "Triad" : memory_mode == COPY ? "Copy" : memory_mode == SCALE ? "Scale" : "Add");
			}

        	printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n","Size (Bytes)","Communication(us)","Kernel(us)","Actual Ratio %","Requested Ratio %","Overall","Overlapping %");
    }
	MPI_Request *reqs=(MPI_Request*)malloc(2*num_neighbors*sizeof(MPI_Request));
    for (long local_N=min_bytes;local_N <= max_bytes; local_N *= 2){
        char *send_buffers[6];
        char *recv_buffers[6];
        	
		for (int i = 0; i < 6; i++) {
			CHECK_CUDA_ERROR(cudaMalloc((void**)&send_buffers[i],local_N));
			CHECK_CUDA_ERROR(cudaMalloc((void**)&recv_buffers[i],local_N));
        	CHECK_CUDA_ERROR(cudaMemset(send_buffers[i],1,local_N));
			CHECK_CUDA_ERROR(cudaMemset(recv_buffers[i],0,local_N));
        }
		for( iter=0;iter<MAX_ITER;iter++){
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			int req_count=0;

			double init_time=MPI_Wtime();
			post_sendrecv(left,right,front,back,bottom,top,dim,send_buffers,recv_buffers,reqs,&req_count,local_N);	
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

            if(iter>=SKIP) {
                t_pure_total += MPI_Wtime()-init_time;
            }
            		
        }
        t_pure_total = 1e6 * t_pure_total/(MAX_ITER-SKIP);
		MPI_Allreduce(&t_pure_total, &t_pure_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			
		for (iter = 0; iter < MAX_ITER; iter++) {
			cudaDeviceSynchronize();
            int req_count=0;
			MPI_Barrier(MPI_COMM_WORLD);

            double init_time = MPI_Wtime();
            post_sendrecv(left,right,front,back,bottom,top,dim,send_buffers,recv_buffers,reqs,&req_count,local_N);
            	
            double targetComputeTime = (compToPureCommRatio/100.0)*t_pure_global;
            t_comp = compute_on_gpu(d_a,stream,grid,block,VECTOR_DIM_COMP,targetComputeTime,measured_unit_us,gpu_inner_iters,max_elems,mem_cal,req_count,reqs,do_progress,compute_bound,memory_mode);

            MPI_Waitall(req_count,reqs,MPI_STATUSES_IGNORE);

            t_ovrl=MPI_Wtime()-init_time;

        	if(iter>=SKIP){
                t_comp_total += t_comp;
                t_ovrl_total += t_ovrl;
            }
        }

        t_comp_total = (t_comp_total)/(MAX_ITER-SKIP);
        t_ovrl_total= (1e6*t_ovrl_total)/(MAX_ITER-SKIP);
		
		MPI_Allreduce(&t_comp_total, &t_compute_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&t_ovrl_total, &t_ovrl_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if(rank == 0) {
			double actualRatio=100.0*(t_compute_global/t_pure_global);
			overlap= 100.0 * fmax(0.0,fmin(1.0,(t_pure_global+t_compute_global-t_ovrl_global)/fmin(t_pure_global, t_compute_global)));
            printf("%-20ld%-20.3f%-20.3f%-20.3f%-20d%-20.3f%-20.3f\n",local_N, t_pure_global, t_compute_global,actualRatio,compToPureCommRatio, t_ovrl_global, overlap);
        }
		t_comp_total=0;
		t_ovrl_total=0;
		t_pure_total=0;
		t_pure_global=0;
		t_compute_global=0;
		t_ovrl_global=0;
		overlap=0;

        for (int i = 0; i < 6; i++) {
        	CHECK_CUDA_ERROR(cudaFree(send_buffers[i]));
        	CHECK_CUDA_ERROR(cudaFree(recv_buffers[i]));
        }
	}
	CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    free(reqs);
	free_vector();
	return 0;
}

#endif
