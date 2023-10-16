#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <assert.h>

//ADDITION 
#include <likwid-marker.h>



double get_curr_time(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double calculateChecksum(long datasetSize, const volatile double* vector) {
    double checksum = 0;
    for (long i = 0; i < datasetSize; i++) {
        checksum += vector[i];
    }
    return checksum;
}

// TASK 1.b
double triad (const long N, const long REP, int *numThreads){
    double time_spent = 0.0;
    double begin, end;


// TASK 1.c
#pragma omp parallel
#pragma omp single
{
	*numThreads=omp_get_num_threads();
}

// TASK 1.d
    size_t ALIGNMENT = (size_t) sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    double* a = (double*) aligned_alloc (ALIGNMENT, N * sizeof(double));
    double* b = (double*) aligned_alloc (ALIGNMENT, N * sizeof(double));
    double* c = (double*) aligned_alloc (ALIGNMENT, N * sizeof(double));
    double* d = (double*) aligned_alloc (ALIGNMENT, N * sizeof(double));

// TASK 1.e
#pragma omp parallel for schedule(static)
    for (long j=0; j<N; j++) {
	    a[j] = 0.0;
	    b[j] = 1.0;
	    c[j] = 2.0;
	    d[j] = 3.0;
    }

// TASK 1.f
#pragma omp parallel
{
    for (long i=0; i<REP; i++)
#pragma omp for schedule(static) nowait
        for (long j=0; j<N; j++)
            a[j] = b[j]+c[j]*d[j];
}

// TASK 1.g
    begin = get_curr_time();


#pragma omp parallel 
{
    LIKWID_MARKER_START("triad");
    for (long i=0; i<REP; i++){
#pragma omp for schedule(static) nowait
        for (long j=0; j<N; j++)
            a[j] = b[j]+c[j]*d[j];

    }
    LIKWID_MARKER_STOP("triad");
}


    end = get_curr_time();
    time_spent = end - begin;


// TASK 1.h
    double sum = calculateChecksum(N, a);
    assert (fabs(sum-N*7.0)<0.1);
    
    free(a); free(b); free(c); free(d);

    return time_spent;
}

int main(int argc, char * argv[]) {
    long int N, REP;
    double time_a;
    double performance_a;

    if (argc != 3 && argc != 4){
        printf("The two parameters N and REP need to be provided.\n");
        exit(1);
    }

    char * pEnd;
    N = strtol (argv[1],&pEnd,10);
    if (errno == ERANGE){
        printf("Problem with the first number.");
        exit(2);
    }
    REP = strtol (argv[2],&pEnd,10);
    if (errno == ERANGE){
        printf("Problem with the second number.");
        exit(3);
    }

    fprintf(stderr, "N = %ld and REP = %ld. Performance in MFLOPS.\n", N, REP);

    printf("| %12s | %12s | %12s | %12s |\n","Dataset size", "Threads", "MFLOPS", "Cycles");

    long datasetSize = N;
    double m_flop = 0;
    int threads;
    long cycles;

    LIKWID_MARKER_INIT;
# pragma omp parallel 
{
LIKWID_MARKER_THREADINIT;
LIKWID_MARKER_REGISTER("triad");
}
     
    
    cycles = 1073741824 / datasetSize;
	if(cycles < 8l){
	    cycles = 8l;
	}
        time_a = triad(datasetSize, cycles, &threads);
        m_flop = 2.0 * (double)datasetSize * (double)cycles * 1.0e-6;
        performance_a = m_flop / time_a;

	printf("| %12ld | %12d | %12.2f | %12ld |\n", datasetSize, threads, performance_a, cycles);
        datasetSize *= 2;
    
    LIKWID_MARKER_CLOSE;
    return 0;
}

