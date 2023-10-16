#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1000000

double get_curr_time(){
	struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e+9 + t.tv_nsec;
}

double branch_F(){
	int i;
	int sum = 0;
	double stime = get_curr_time();
	for (i = 0; i < N; i++)
		if (i && 0) // FFFF...
			sum++;
	double dur = get_curr_time() - stime;
	return dur;
}

double branch_T(){
	int i;
	int sum = 0;
	double stime = get_curr_time();
	for (i = 0; i < N; i++)
		if (i || 1) // TTTT...
			sum++;
	double dur = get_curr_time() - stime;
	return dur;
}

double branch_TF(){
	int i;
	int sum = 0;
	double stime = get_curr_time();
	for (i = 0; i < N; i++)
		if (i && (i%2)) // FTFT...
			sum++;
	double dur = get_curr_time() - stime;
	return dur;
}

double branch_rand(){
	int i;
	int sum = 0;
	double stime = get_curr_time();
	for (i = 0; i < N; i++)
		if (rand()%2) // random TF...
			sum++;
	double dur = get_curr_time() - stime;
	return dur;
}

int main(int argc, char * argv[]) {
    
    printf("| %12s | %12s | %12s |\n","Loop", "Branch", "Time(ns)");
    printf("|--------------------------------------------|\n");
    double br_F = branch_F();
    printf("| %12d | %12s | %12.3f |\n", N, "FFFF...", br_F);
    printf("|--------------------------------------------|\n");
    double br_T = branch_T();
    printf("| %12d | %12s | %12.3f |\n", N, "TTTT...", br_T);
    printf("|--------------------------------------------|\n");
    double br_TF = branch_TF();
    printf("| %12d | %12s | %12.3f |\n", N, "TFTF...", br_TF);
    printf("|--------------------------------------------|\n");
    double br_RAND = branch_rand();
    printf("| %12d | %12s | %12.3f |\n", N, "RAND.T.F", br_RAND);

    return 0;
}
