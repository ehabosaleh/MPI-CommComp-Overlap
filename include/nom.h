#ifndef NOM_H
#define NOM_H

#include<mpi.h>

#ifndef MAX_ITER
#define MAX_ITER 200
#endif

#ifndef SKIP
#define SKIP 100
#endif


#ifndef MAX_MESSAGE_SIZE
#define MAX_MESSAGE_SIZE (1 << 28)
#endif

#ifndef MIN_MESSAGE_SIZE
#define MIN_MESSAGE_SIZE (1 << 0)
#endif


int run_overlap_benchmark(int rank,int size);

#endif

