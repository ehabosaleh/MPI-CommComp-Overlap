#include<stdlib.h>
#include<mpi.h>
#include"compute.h"


float **a=NULL;
float *x=NULL;
float *y=NULL;

void init_arrays(void){
	int i,j;
	a=(float**)malloc(DIM*sizeof(float*));
	for(i=0;i<DIM;i++)
		a[i]=(float*)malloc(DIM*sizeof(float));

	x=(float*)malloc(DIM*sizeof(float));
	y=(float*)malloc(DIM*sizeof(float));

	for(i=0;i<DIM;i++){
		x[i]=1;
		y[i]=1;

		for(j=0;j<DIM;j++){
			a[i][j]=2;
		}
	
	}

}

void compute_on_host(double latency){
	int i,j;
	double ccompute_start=0;
	double ccompute_total=0;
	
	while(ccompute_total<latency){
		ccompute_start=MPI_Wtime();
		for(i=0;i<DIM;i++)
			for(j=0;j<DIM;j++)
				x[i]=x[i]+a[i][j]*a[i][j]+y[j];

		ccompute_total+=MPI_Wtime()-ccompute_start;
	}


}

