#include<stdlib.h>
#include<mpi.h>
#include"compute.h"


float *a=NULL;
float *x=NULL;
float *y=NULL;

void init_arrays(void){
	int i,j;
	a=(float*)malloc(ARRAY_DIM*ARRAY_DIM*sizeof(float));

	x=(float*)malloc(ARRAY_DIM*sizeof(float));
	y=(float*)malloc(ARRAY_DIM*sizeof(float));

	for(i=0;i<ARRAY_DIM;i++){
		x[i]=1;
		y[i]=1;
	}

	for(j=0;j<ARRAY_DIM*ARRAY_DIM;j++){
		a[j]=2;
	}
	

}
void compute_on_host(double latency){
	int i,j;

	double sum=0;
	double ccompute_start=MPI_Wtime();
	double ccompute_end=ccompute_start;
	while((ccompute_end-ccompute_start)<latency){
		for(i=0;i<ARRAY_DIM;i++)
			for(j=0;j<ARRAY_DIM;j++)
				x[i]=x[i]+A*a[i*ARRAY_DIM+j]*y[j];
		ccompute_end=MPI_Wtime();
	}
}

