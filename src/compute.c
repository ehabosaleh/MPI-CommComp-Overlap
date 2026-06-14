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

static void compute_bound_step(const long N){
	double s0=0.0, s1=0.0, s2=0.0, s3=0.0;
	for(int i=0;i<N;i++){
		s0=s0*1.0000001+s1;
		s1=s1*1.0000002+s2;
		s2=s2*1.0000003+s3;
		s3=s3*1.0000003+s0;
	}
	
}

void compute_on_host(double latency){
	int i,j;
	double ccompute_start=0;
	double ccompute_total=0;
	
	while(ccompute_total<latency){
		ccompute_start=MPI_Wtime();
		
		for(i=0;i<ARRAY_DIM;i++)
			for(j=0;j<ARRAY_DIM;j++)
				x[i]=x[i]+a[i*ARRAY_DIM+j]*a[i*ARRAY_DIM+j]+y[j];
		

		ccompute_total+=MPI_Wtime()-ccompute_start;
	}


}

