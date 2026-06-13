#include"nmpm.h"
#include"compute.h"
void usage(char *prog_name) {
	fprintf(stderr, "Usage: %s [--dim=N] [--ratio=P]\n", prog_name);
	fprintf(stderr, "  <dim>: 2 for 2D grid, 3 for 3D grid\n");
	fprintf(stderr, "  <ratio>: Desired computation to pure communication time ratio (e.g., 50 for 50%%)\n");
}
int run_overlap_benchmark(int rank, int size, int dim, int compToPureCommRatio){
		int iter;
    	double t_pure=0, t_pure_total=0;
    	double tcomp=0, tcomp_total=0;
    	double t_ovrl=0, t_ovrl_total=0;
    	double overlap=0, overlap_avr=0;
    	double t_pure_total0=0, tcomp_total0=0, t_ovrl_total0=0;
		int dims[3], coords[3];
		int is_3d=0;
		int num_neighbors;

		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    	MPI_Comm_size(MPI_COMM_WORLD,&size);
		if(dim==3){

			int cbrt_size = (int)cbrt((double)size);
			if (cbrt_size * cbrt_size * cbrt_size != size) {
				if (rank == 0) {
					fprintf(stderr, "Number of processes must be a perfect cube for 3D grid\n");
				}
				MPI_Finalize();
				return -1;
			}

			is_3d = 1;
			dims[0] = dims[1] = dims[2] = cbrt_size;
			coords[0] = rank / (cbrt_size * cbrt_size);
			coords[1] = (rank % (cbrt_size * cbrt_size)) / cbrt_size;
			coords[2] = rank % cbrt_size;
			num_neighbors = 6;
			}
		else if(dim==2){
			int sqrt_size = (int)sqrt((double)size);
			if (sqrt_size * sqrt_size != size) {
				if (rank == 0) {
					fprintf(stderr, "Number of processes must be a perfect square or perfect cube\n");
				}
				MPI_Finalize();
				return -1;
			}
			dims[0] = dims[1] = sqrt_size;
			dims[2] = 1;
			coords[0] = rank / sqrt_size;
			coords[1] = rank % sqrt_size;
			coords[2] = 0;
			num_neighbors = 4;
		}

    	init_arrays();

    	if (rank==0) {
			if (is_3d) {
				printf("Running 3D benchmark with grid %dx%dx%d\n", dims[0], dims[1], dims[2]);
			} else {
				printf("Running 2D benchmark with grid %dx%d\n", dims[0], dims[1]);
			}
        	printf("%-20s%-20s%-20s%-20s%-20s%-20s\n","Size (Bytes)","Communication(us)","Computation(us)","CompToCommRatio %","Overall","Overlapping %");
    	}

    	MPI_Request *reqs=(MPI_Request*)malloc(2*num_neighbors*sizeof(MPI_Request));
    	for (long local_N=MIN_MESSAGE_SIZE;local_N <= MAX_MESSAGE_SIZE; local_N *= 2) {

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

        	t_pure=0;
        	t_pure_total=0;
        	tcomp_total=0;
        	t_ovrl_total=0;
        	for (iter =0; iter< MAX_ITER; iter++){
            		int req_count=0;
            		double init_time=MPI_Wtime();
            		if (is_3d) {
						int left = (coords[2]==0) ? MPI_PROC_NULL : rank - 1;
						int right = (coords[2]==dims[2]-1) ? MPI_PROC_NULL : rank + 1;
						int front = (coords[1]==0) ? MPI_PROC_NULL : rank - dims[2];
						int back = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + dims[2];
						int bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[2]*dims[1];
						int top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[2]*dims[1];

						if(left != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(right != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(front != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(back != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(top != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(bottom != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
            		} else {
						int left = (coords[1]==0) ? MPI_PROC_NULL : rank - 1;
						int right = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + 1;
						int top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[1];
						int bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[1];

						if(left != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(right != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(top != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(bottom != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
            		}

            		MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

            		if(iter>SKIP) {
                		t_pure += MPI_Wtime() - init_time;
            		}
            		MPI_Barrier(MPI_COMM_WORLD);
        	}

        	MPI_Barrier(MPI_COMM_WORLD);
        	t_pure_total = 1e6 * t_pure / (MAX_ITER - SKIP);


        	for (iter = 0; iter < MAX_ITER; iter++) {
            		int req_count = 0;
            		double init_time = MPI_Wtime();
            		
            		if (is_3d) {
						int left = (coords[2]==0) ? MPI_PROC_NULL : rank - 1;
						int right = (coords[2]==dims[2]-1) ? MPI_PROC_NULL : rank + 1;
						int front = (coords[1]==0) ? MPI_PROC_NULL : rank - dims[2];
						int back = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + dims[2];
						int bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[2]*dims[1];
						int top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[2]*dims[1];

						if(left != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(right != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(front != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, front, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(back != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, back, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(top != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[4], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(bottom != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[5], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
            		} else {
						int left = (coords[1]==0) ? MPI_PROC_NULL : rank - 1;
						int right = (coords[1]==dims[1]-1) ? MPI_PROC_NULL : rank + 1;
						int top = (coords[0]==0) ? MPI_PROC_NULL : rank - dims[1];
						int bottom = (coords[0]==dims[0]-1) ? MPI_PROC_NULL : rank + dims[1];

						if(left != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[0], local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(right != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[1], local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(top != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[2], local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
						if(bottom != MPI_PROC_NULL) {
							MPI_Isend(send_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
							MPI_Irecv(recv_buffers[3], local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
						}
            		}
            		double targetComputeTime = (compToPureCommRatio/100.0)*t_pure_total;
            		double tcomp_start = MPI_Wtime();
            		compute_on_host((targetComputeTime/1e6));
            		tcomp = MPI_Wtime()-tcomp_start;

            		MPI_Waitall(req_count,reqs,MPI_STATUSES_IGNORE);

            		t_ovrl=MPI_Wtime()-init_time;

            		if(iter>SKIP) {
                		tcomp_total += tcomp;
                		t_ovrl_total += t_ovrl;
            		}
            		MPI_Barrier(MPI_COMM_WORLD);
        	}

        	tcomp_total = (tcomp_total*1e6)/(MAX_ITER-SKIP);
        	t_ovrl_total= (t_ovrl_total*1e6)/(MAX_ITER-SKIP);

        	overlap = 100.0 * fmax(0.0,fmin(1.0,(t_pure_total+tcomp_total-t_ovrl_total)/fmin(t_pure_total, tcomp_total)));

        	MPI_Reduce(&overlap,&overlap_avr,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
        	MPI_Reduce(&t_ovrl_total,&t_ovrl_total0,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
        	MPI_Reduce(&tcomp_total,&tcomp_total0, 1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
       		MPI_Reduce(&t_pure_total,&t_pure_total0,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

        	if(rank == 0) {
            		t_pure_total0/= size;
            		tcomp_total0 /= size;
            		t_ovrl_total0/= size;
            		overlap_avr/= size;

            		printf("%-20ld%-20.3f%-20.3f%-20d%-20.3f%-20.3f\n",local_N, t_pure_total0, tcomp_total0,compToPureCommRatio, t_ovrl_total0, overlap_avr);
        	}

        	overlap_avr=0;
        	tcomp_total=0;
        	t_ovrl_total=0;
        	t_pure_total0=0;
        	tcomp_total0=0;
        	t_ovrl_total0=0;

        	for (int i = 0; i < 6; i++) {
        		free(send_buffers[i]);
        		free(recv_buffers[i]);
        	}
    	}

    	free(reqs);
	return 0;
}

