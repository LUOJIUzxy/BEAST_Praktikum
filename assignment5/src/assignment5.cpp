// put your working code in here to use the test_script.sh and CI

#include <chrono>
#include <math.h>
#include <cstdio>
#include <omp.h>
// use a 1d array to guarantee a dense and sequential memory layout
#define TWO_D_ACCESS(row, col, width) ((width) * (row) + (col))


void mm_kernel(double* a, double* b, double* c, int N, int numTeams, int numThreads) {

    #pragma omp target 
    #pragma omp teams num_teams(numTeams) thread_limit(numThreads)
    #pragma omp distribute parallel for schedule(static,1) collapse(2) num_threads(numThreads)
    for( long i = 0; i < N; ++i ) {
        for( long j = 0; j < N; ++j ) {
            for( int k = 0; k < N; ++k ) {

                a[TWO_D_ACCESS(i, j, N)] += b[TWO_D_ACCESS(i, k, N)] * c[TWO_D_ACCESS(j, k, N)];
            }
        }
    }
}

double mm(int N, int REP, double expected_result, int numTeams, int numThreads) {

  int data_size = N * N;
  double * a = (double *) malloc(N*N*sizeof(double));
  double * b = (double *) malloc(N*N*sizeof(double));
  double * c = (double *) malloc(N*N*sizeof(double));

  //Set GPU data scope and allocate memory 
  #pragma omp target enter data map(alloc: a[0:data_size], b[0:data_size], c[0:data_size]) 

  #pragma omp target
  #pragma omp teams num_teams(numTeams) thread_limit(numThreads)
  #pragma omp distribute parallel for num_threads(numThreads)
  for(long i = 0; i < N; ++i) {
      for(int j = 0; j < N; ++j) {
        a[i*N + j] = 0;
        // initialize arrays with something so the compiler/cpu cannot optimize 0*0=0
        /*
        * TODO : Task 2 : Change initializations here to apply column and row major storage as asked in task sheet
        */
        b[i*N + j] = (i*j)/(N*50); 
        c[i*N + j] = (i*j)/(N*50);  
      }
  }

  //Start time measurements for computation
  auto t0 = std::chrono::high_resolution_clock::now();

  for( int r=0; r<REP; ++r ) {
  mm_kernel(a, b, c, N, numTeams, numThreads);
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  //Exit GPU data scope
  #pragma omp target exit data map(from: a[0:data_size]) map(release: b[0:data_size], c[0:data_size]) 
    

  // simple correctness check
  double array_sum = 0;
  for( int i=0; i<N*N; ++i ) {
    array_sum += a[i];
  }

  // verify expected result. accounting for possibly system dependent float rounding errors
  if(abs(array_sum - expected_result) > 0.001){
    printf("Wrong result for N=%4d. expected %.3f but got %.3f. Aborting...\n", N, expected_result, array_sum);
    exit(EXIT_FAILURE);
  }

  // Free memory
  free((void *) a);
  free((void *) b);
  free((void *) c);

  //Time calculation
  using dsec = std::chrono::duration<double>;
  double dur = std::chrono::duration_cast<dsec>(t1-t0).count();

  //Flop calculation
  double mflop = 2.0*(double)N*(double)N*(double)N*(double)REP*1.0e-6;
  return mflop/dur;
}

int main(int argc, char* argv[]) {

  
  double mf;
  
  /*
    * TODO: Task 3: You can adjust thread and team numbers by passing argv[1] and argv[2] to program
    */ 
  char *pEnd;
  long numTeams=0;
  long numThreads=0;
  if (argc >= 3){
    numTeams = strtol(argv[1], &pEnd ,10);
    numThreads= strtol(argv[2],&pEnd, 10);
  }
  else{
   printf("Wrong usage of the template. League size and Team size should be passed !\n ");
   return 0;
  }

  // ugly hardcoded stuff :)
  mf = mm(100, 2000, 108290000.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 100, mf);

  mf = mm(200, 250, 1170799500.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 200, mf);

  mf = mm(300, 75, 3560702175.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 300, mf);

  mf = mm(400, 32, 7353887776.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 400, mf);

  mf = mm(500, 16, 12171228800.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 500, mf);

  mf = mm(750, 10, 64265028760.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 750, mf);

  mf = mm(1000, 6, 171249832524.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 1000, mf);

  mf = mm(1250, 4, 359491225496.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 1250, mf);

  mf = mm(1500, 3, 685019722065.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 1500, mf);

  mf = mm(1750, 2, 1001806087176.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 1750, mf);

  mf = mm(2000, 2, 1974958963390.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 2000, mf);

  mf = mm(2500, 1, 3060585851209.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 2500, mf);

  mf = mm(3000, 1, 7694687665084.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 3000, mf);

  mf = mm(3500, 1, 16754058063962.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 3500, mf);

  mf = mm(4000, 1, 32845267978538.000, numTeams, numThreads);
  printf("MFLOPS(N=%4d): %.3f\n", 4000, mf);

  return 0;
}
