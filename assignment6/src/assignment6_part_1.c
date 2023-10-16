#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <algorithm>

#include <omp.h>

double calculateMegaFlopRate(long size, long repetitions, double duration) {
    /*
     * TODO@Students: Q1a) Calculate MegaFLOP rate
     */
}

double calculateChecksum(long datasetSize, const volatile double* vector, int stride) {
    double checksum = 0;
    for (int i = 0; i < datasetSize; i+=stride) {
        checksum += vector[i];
    }
    return checksum;
}

void triad(long datasetSize, long repetitions, long numThreads, int stride) {

    /*
    * TODO@Students: Q1a) Add your parallel solution for triad benchmark implementation from assignment 1 
    * TODO@Students: Q1a) Increment the interator of the inner for loop - where you do triad computations - in step sizes of stride
    */

    double mflops = calculateMegaFlopRate(datasetSize, repetitions, duration);
    printf("| %10ld | %8d | %8.2f | %8ld | %.4e |\n", datasetSize, stride, mflops, repetitions, checksum);
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("The two parameters maximum dataset size and total number of processed points need to be provided.\n");
        exit(1);
    }

    char *pEnd;
    long maximumDatasetSize = strtol(argv[1], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the first number.");
        exit(2);
    }
    long totalNumberProcessedPoints = strtol(argv[2], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the second number.");
        exit(3);
    }

    fprintf(
            stderr, "Maximum dataset size = %ld, total number of processed points = %ld. Performance in MFLOPS.\n",
            maximumDatasetSize, totalNumberProcessedPoints
    );
    printf("| %10s | %8s | %8s | %8s | %10s |\n", "Data size", "Stride", "Access Rate", "Cycles", "Checksum");

    /*
    * TODO@Students: Q1b) Try all different stride values from the set {1, 2, 4, 8, 16, 32, 64} 
    */
    long  datasetSize = 64;
    while (datasetSize <= maximumDatasetSize) {
        // Keep the total number of processed points constant by adjusting the number of repetitions according to data
        // set size
        long cycles = std::clamp(totalNumberProcessedPoints / datasetSize, 8l, 65536l);
        long threads = omp_get_max_threads();
        triad(datasetSize, cycles, threads, stride);
            
        datasetSize *= 2;  
    }
    return 0;
}
