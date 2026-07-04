#include<stdlib.h>
#include<mpi.h>
#include"compute_cpu.h"

double *mb_a = NULL;
double *mb_b = NULL;
double *mb_c = NULL;
size_t mb_elems = 0;


int init_memory_bound_buffers(size_t bytes){
	if (bytes < SIZE_THRESHOLD)
        bytes = SIZE_THRESHOLD;
		
	mb_elems=bytes/sizeof(double);

    if(posix_memalign((void **)&mb_a, 64, mb_elems * sizeof(double)) != 0)
		return -1;

    if(posix_memalign((void **)&mb_b, 64, mb_elems * sizeof(double)) != 0)
        return -1;

    if(posix_memalign((void **)&mb_c, 64, mb_elems * sizeof(double)) != 0)
        return -1;

    for(size_t i=0; i<mb_elems;i++) {
        mb_a[i]=1.0;
        mb_b[i]=2.0;
        mb_c[i]=0.0;
    }

    return 0;
}

void free_memory_bound_buffers(void)
{
    free(mb_a);
    free(mb_b);
    free(mb_c);

    mb_a = NULL;
    mb_b = NULL;
    mb_c = NULL;
    mb_elems = 0;
}
/*
void compute_on_host(double latency, int size_threshold){
	int i,j;

	double ccompute_start=MPI_Wtime();
	double ccompute_end=ccompute_start;
	if(size_threshold>=SIZE_THRESHOLD)
		while((ccompute_end-ccompute_start)<latency){
				for(i=0;i<ARRAY_DIM;i++)
					for(j=0;j<ARRAY_DIM;j++)
						x[i]=x[i]+A*a[i*ARRAY_DIM+j]*y[j];
				ccompute_end=MPI_Wtime();
		}
	else{
		while((ccompute_end-ccompute_start)<latency){
				x[0]=x[0]+A*a[0]*y[0];
				ccompute_end=MPI_Wtime();
		}
	}
}
*/

NOINLINE void cpu_compute_bound_batch(void){
    double r0=x[0]+1.0e-30*host_sink;
    double r1=x[1]+y[0];
    double r2=x[2]+y[1];
    double r3=x[3]+y[2];

    double r4=a[0]+0.1;
    double r5=a[1]+0.2;
    double r6=a[2]+0.3;
    double r7=a[3]+0.4;

    for (int k = 0; k < COMPUTE_INNER_ITERS; k++) {
        r0 = r0 * 1.000000000000001 + r4;
        r1 = r1 * 1.000000000000002 + r5;
        r2 = r2 * 0.999999999999999 + r6;
        r3 = r3 * 1.000000000000003 + r7;

        r4 = r4 * 1.000000000000004 + 0.000000000000001;
        r5 = r5 * 0.999999999999998 + 0.000000000000002;
        r6 = r6 * 1.000000000000005 + 0.000000000000003;
        r7 = r7 * 0.999999999999997 + 0.000000000000004;
    }

    x[0] = r0;
    x[1] = r1;
    x[2] = r2;
    x[3] = r3;

    host_sink += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
}
NOINLINE void cpu_memory_bound_batch(void){
    if (mb_a==NULL||mb_b==NULL||mb_c==NULL||mb_elems==0) {
        fprintf(stderr, "Memory-bound buffers are not initialized.\n");
        return;
    }

    for (size_t i = 0; i < mb_elems; i++) {
        mb_c[i] = mb_a[i] + A * mb_b[i];
    }

    host_sink +=mb_c[mb_elems-1];
}
void compute_on_host(double latency_sec,int compute_bound){
    if(latency_sec<MIN_COMPUTE_SEC)
        latency_sec=MIN_COMPUTE_SEC;

    double start=MPI_Wtime();
    double now=start;

    int check_interval;

    if(latency_sec <= 10.0e-6)
        check_interval = TIME_CHECK_INTERVAL_SHORT;
    else
        check_interval = TIME_CHECK_INTERVAL_LONG;

    do{
        for (int r = 0; r < check_interval; r++) {
            if (compute_bound) {
                cpu_compute_bound_batch();
            } else {
                cpu_memory_bound_batch();
            }
        }

        now = MPI_Wtime();

    } while ((now - start) < latency_sec);
}