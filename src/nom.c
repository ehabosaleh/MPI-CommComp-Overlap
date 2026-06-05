#include"nom.h"
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<compute.h>

int run_overlap_benchmark(int rank, int size){
	int iter;
    	double t_pure=0, t_pure_total=0;
    	double tcomp=0, tcomp_total=0;
    	double t_ovrl=0, t_ovrl_total=0;
    	double overlap=0, overlap_avr=0;
    	double t_pure_total0=0, tcomp_total0=0, t_ovrl_total0=0;
	int left=0,right=0,top=0,bottom=0;
	int dims[2], coords[2];

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    	MPI_Comm_size(MPI_COMM_WORLD,&size);

    	int sqrt_size = (int)sqrt((double)size);
    	if (sqrt_size * sqrt_size != size) {
        	if (rank == 0) {
            		fprintf(stderr, "Number of processes must be a perfect square\n");
        	}
        MPI_Finalize();
        return -1;
    	}

    	dims[0]=dims[1]=sqrt_size;

    	coords[0]=rank/sqrt_size;
    	coords[1]=rank%sqrt_size;

    	left=(coords[1]==0)? MPI_PROC_NULL:rank-1;
    	right=(coords[1]==dims[1]-1)?MPI_PROC_NULL:rank+1;
    	top=(coords[0]==0)? MPI_PROC_NULL:rank-sqrt_size;
    	bottom=(coords[0]==dims[0]-1)?MPI_PROC_NULL:rank + sqrt_size;

    	init_arrays();

    	if (rank==0) {
        	printf("%-20s%-20s%-20s%-20s%-20s\n","Size (Bytes)","Communication(us)","Computation(us)","Overall","Overlapping %");
    	}

    	/* At most 8 requests (4 neighbors * 2) */
    	MPI_Request *reqs=(MPI_Request*)malloc(2*size*sizeof(MPI_Request));
    	for (long local_N=MIN_MESSAGE_SIZE;local_N <= MAX_MESSAGE_SIZE; local_N *= 2) {

        	char *u_0      = (char *)malloc(local_N);
        	char *u_1      = (char *)malloc(local_N);
        	char *u_2      = (char *)malloc(local_N);
        	char *u_3      = (char *)malloc(local_N);
        	char *u_left   = (char *)malloc(local_N);
        	char *u_right  = (char *)malloc(local_N);
        	char *u_top    = (char *)malloc(local_N);
        	char *u_bottom = (char *)malloc(local_N);

        	for(long i=0;i<local_N;i++) {
            		u_0[i]      = 'a';
            		u_1[i]      = 'b';
            		u_2[i]      = 'c';
            		u_3[i]      = 'd';
            		u_left[i]   = 'f';
            		u_right[i]  = 'g';
            		u_top[i]    = 'h';
            		u_bottom[i] = 'i';
        	}

        	t_pure=0;
        	t_pure_total=0;
        	tcomp_total=0;
        	t_ovrl_total=0;

        	/* Phase-1: Pure communication time */
        	for (iter =0;iter< MAX_ITER;iter++){
            		int req_count=0;
            		double init_time=MPI_Wtime();

            		if(left !=MPI_PROC_NULL) {
                		MPI_Isend(u_0, local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_left, local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(right != MPI_PROC_NULL) {
                		MPI_Isend(u_1, local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_right, local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(top != MPI_PROC_NULL) {
                		MPI_Isend(u_2, local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_top, local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(bottom!= MPI_PROC_NULL) {
                		MPI_Isend(u_3, local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_bottom, local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

            		if(iter>SKIP) {
                		t_pure += MPI_Wtime() - init_time;
            		}
            		MPI_Barrier(MPI_COMM_WORLD);
        	}

        	MPI_Barrier(MPI_COMM_WORLD);
        	t_pure_total = 1e6 * t_pure / (MAX_ITER - SKIP);


        	/* Phase-2:  communication + computation (overlap) */
        	for (iter = 0; iter < MAX_ITER; iter++) {
            		int req_count = 0;
            		double init_time = MPI_Wtime();
            		if(left != MPI_PROC_NULL) {
                		MPI_Isend(u_0, local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_left, local_N, MPI_CHAR, left, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(right!= MPI_PROC_NULL) {
                		MPI_Isend(u_1, local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_right, local_N, MPI_CHAR, right, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(top!=MPI_PROC_NULL) {
                		MPI_Isend(u_2, local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_top, local_N, MPI_CHAR, top, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}

            		if(bottom!= MPI_PROC_NULL) {
                		MPI_Isend(u_3, local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                		MPI_Irecv(u_bottom, local_N, MPI_CHAR, bottom, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            		}
            		double tcomp_start = MPI_Wtime();
            		compute_on_host(t_pure_total/1e6);
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

        	/* Overlap ratio */
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

            		printf("%-20ld%-20.3f%-20.3f%-20.3f%-20.3f\n",local_N, t_pure_total0, tcomp_total0, t_ovrl_total0, overlap_avr);
        	}

        	overlap_avr=0;
        	tcomp_total=0;
        	t_ovrl_total=0;
        	t_pure_total0=0;
        	tcomp_total0=0;
        	t_ovrl_total0=0;

        	free(u_0);
        	free(u_1);
        	free(u_2);
        	free(u_3);
        	free(u_left);
        	free(u_right);
        	free(u_top);
        	free(u_bottom);
    	}

    	free(reqs);
	return 0;
}

