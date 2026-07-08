#include"compute_cpu.h"


double x[ARRAY_DIM];
double a[ARRAY_DIM * ARRAY_DIM];
double y[ARRAY_DIM];

double *mb_a=NULL;
double *mb_b=NULL;
double *mb_c=NULL;
size_t mb_elems=0;

volatile double host_sink=0.0;

static size_t mb_offset=0;

int init_memory_bound_buffers(size_t bytes){
		
	mb_elems=bytes/sizeof(double);

    if(posix_memalign((void **)&mb_a,64,mb_elems * sizeof(double)) != 0)
		return -1;

    if(posix_memalign((void **)&mb_b,64,mb_elems*sizeof(double)) != 0)
        return -1;

    if(posix_memalign((void **)&mb_c,64,mb_elems*sizeof(double)) != 0)
        return -1;

    for(size_t i=0; i<mb_elems;i++) {
        mb_a[i]=1.0;
        mb_b[i]=2.0;
        mb_c[i]=0.0;
    }

    return 0;
}

void free_memory_bound_buffers(void){
    free(mb_a);
    free(mb_b);
    free(mb_c);

    mb_a = NULL;
    mb_b = NULL;
    mb_c = NULL;
    mb_elems = 0;
}

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
NOINLINE void cpu_memory_bound_batch(memory_mode_t memory_mode){
    if (mb_a == NULL || mb_b == NULL || mb_c == NULL || mb_elems == 0) {
        fprintf(stderr, "Memory-bound buffers are not initialized.\n");
        return;
    }

    size_t chunk = MEMORY_CHUNK_ELEMS;

    if (chunk > mb_elems)
        chunk = mb_elems;

    if (mb_offset + chunk > mb_elems)
        mb_offset = 0;

    size_t start = mb_offset;
    size_t end   = mb_offset + chunk;
    switch (memory_mode) {
        case MEMORY_MODE_TRIAD:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i] + A * mb_b[i];
            }
            break;
        case MEMORY_MODE_COPY:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i];
            }
            break;
        case MEMORY_MODE_SCALE:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = A * mb_a[i];
            }
            break;
        case MEMORY_MODE_ADD:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i] + mb_b[i];
            }
            break;
        default:
            fprintf(stderr, "Invalid memory mode specified.\n");
            return;
    }

    mb_offset += chunk;

    host_sink += mb_c[end - 1];
}
void * progress_thread_func(void *arg) {
    progress_thread_data_t *req = (progress_thread_data_t *)arg;
    /*
    while (!req->stop_flag) {
        MPI_Testall(req->num_requests, req->requests, &req->stop_flag, MPI_STATUSES_IGNORE);
    }
    */
    while(!atomic_load_int(&req->terminate)) {
        if (!atomic_load_int(&req->active)) {
            sched_yield();
            continue;
        }

        while (!req->stop_flag) {
            MPI_Testall(req->num_requests, req->requests, &req->stop_flag, MPI_STATUSES_IGNORE);
        }
        req->stop_flag = 0;
        //MPI_Waitall(req->num_requests, req->requests, MPI_STATUSES_IGNORE);
        atomic_store_int(&req->done, 1);
        atomic_store_int(&req->active, 0);

    }    
    return NULL;
}
int start_progress_thread(progress_thread_data_t *progress_data) {
    progress_data->requests = NULL;
    progress_data->num_requests = 0;
    if(progress_data->is_gpu) {
        printf("Cuda device was set for dev %d\n",progress_data->cuda_device);
        cudaSetDevice(progress_data->cuda_device);
    } else {
        progress_data->cuda_device = -1;
    }
    atomic_store_int(&progress_data->active, 0);
    atomic_store_int(&progress_data->done, 1);
    atomic_store_int(&progress_data->terminate, 0);

    int rc = pthread_create(&progress_data->thread,NULL,progress_thread_func,progress_data);
    if (rc != 0) {
        fprintf(stderr, "Error creating progress thread\n");
        return -1;
    }

    return 0;
}
int post_progress_thread_requests(progress_thread_data_t *progress_data, MPI_Request *requests, int num_requests) {
    if(atomic_load_int(&progress_data->active)) {
        fprintf(stderr, "Progress thread is already active\n");
        return -1;
    }
    progress_data->requests = requests;
    progress_data->num_requests = num_requests;

    atomic_store_int(&progress_data->done, 0);
    atomic_store_int(&progress_data->active, 1);

    return 0;
}
int wait_progress_thread(progress_thread_data_t *progress_data) {
    while(!atomic_load_int(&progress_data->done)) {
        sched_yield();
    }
    return 0;
}
int terminate_progress_thread(progress_thread_data_t *progress_data) {
    wait_progress_thread(progress_data);
    atomic_store_int(&progress_data->terminate, 1);
    pthread_join(progress_data->thread, NULL);
    return 0;
}
void compute_on_host(double latency_sec,int compute_bound,memory_mode_t memory_mode) {
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
    
                cpu_memory_bound_batch(memory_mode);
            }
        }

        now = MPI_Wtime();

    } while ((now - start) < latency_sec);
}