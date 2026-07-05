#include<stdlib.h>

#define ARRAY_DIM 25
#define COMPUTE_INNER_ITERS 128

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif


double x[ARRAY_DIM];
double a[ARRAY_DIM * ARRAY_DIM];
double y[ARRAY_DIM];
double host_sink=0.0;

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
        r0=r0*1.000000000000001+r4;
        r1=r1*1.000000000000002+r5;
        r2=r2*0.999999999999999+r6;
        r3=r3*1.000000000000003+r7;

        r4=r4*1.000000000000004+0.000000000000001;
        r5=r5*0.999999999999998+0.000000000000002;
        r6=r6*1.000000000000005+0.000000000000003;
        r7=r7*0.999999999999997+0.000000000000004;
    }

    x[0]=r0;
    x[1]=r1;
    x[2]=r2;
    x[3]=r3;

    host_sink+=r0+r1+r2+r3+r4+r5+r6+r7;
}

int main(void){
	for(int i=0;i<10000000;i++)
		cpu_compute_bound_batch();
}
