#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <algorithm>

#include <omp.h>

double calculateMegaFlopRate(long size, long repetitions, double duration) {
    return 2.0 * (double) size * (double) repetitions * 1.0e-6 / duration;
}

double calculateChecksum(long datasetSize, const volatile double* vector) {
    double checksum = 0;
    for (int i = 0; i < datasetSize; i++) {
        checksum += vector[i];
    }
    return checksum;
}

void triad(unsigned long datasetSize, unsigned long repetitions, long numTeams, long numThreads) {

    volatile auto *a = new double[datasetSize];
    auto *b = new double[datasetSize];
    auto *c = new double[datasetSize];
    auto *d = new double[datasetSize];

    // Device Ids CPU = 1 , GPU = 0 
    int targetDeviceId = 1;

    /*
    *TODO: Task 1 - Check by below code block to see if you can offload to GPU properly
    */
    #pragma omp target map(tofrom: targetDeviceId, numTeams, numThreads)
    {
      targetDeviceId = omp_is_initial_device();
    }

    /*
    *TODO: Task 3: Variant 1 (CPU to GPU copy after CPU initialization ) is done by the for loop below. Implement and Test Variant 2 
    */
#pragma omp target enter data map(alloc: a[0:datasetSize]) map(to: b[0:datasetSize], c[0:datasetSize], d[0:datasetSize])
    #pragma omp target map(tofrom: a[0:datasetSize]) map(to: b[0:datasetSize], c[0:datasetSize], d[0:datasetSize])
    for (unsigned long i = 0; i < datasetSize; i++) {
        a[i] = b[i] = c[i] = d[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    //GPU offloading o f the computation loops 
    #pragma omp target map(tofrom: a[0:datasetSize]) map(to: b[0:datasetSize], c[0:datasetSize], d[0:datasetSize])
    #pragma omp teams num_teams(numTeams) thread_limit(numThreads)
    for (unsigned long j = 0; j < repetitions; ++j) {

        /*
        *TODO: Task 4: Add necessary scheduling clause to the pragma below. 
        *             (You can play and test with different scheduling schemes and chunk sizes to better see the effects)
        */
        /*
        *TODO: Task 2: Add necessary clause to the pragma below to utilise all threads and teams to full extend
        *             (You can play and test with different scheduling schemes and chunk sizes to better see the effects)
        */
        #pragma omp distribute parallel for num_threads(numThreads)
        for (unsigned long i = 0; i < datasetSize; ++i) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
   #pragma omp target exit data map(from: a[0:datasetSize]) map(release:b[0:datasetSize], c[0:datasetSize],d[0:datasetSize])


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(stop - start).count();

    double checksum = calculateChecksum(datasetSize, a);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;

    double mflops = calculateMegaFlopRate(datasetSize, repetitions, duration);
    printf("| %10ld | %8ld | %8ld | %8.2f | %8ld | %4d | %.4e |\n", datasetSize, numTeams, numThreads, mflops, repetitions, targetDeviceId, checksum);
}

int main(int argc, char *argv[]) {

   if (argc < 3) {
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
    /*
    * TODO: Task 5: You can adjust thread and team numbers by passing argv[3] and argv[4] to program
    */ 
    long numThreads = 32;
    long numTeams = 15;
    if(argc >= 4 )
        numTeams = strtol(argv[3], &pEnd, 10);
    if(argc == 5 )
        numThreads = strtol(argv[4], &pEnd, 10);

    fprintf(
            stderr, "Maximum dataset size = %ld, total number of processed points = %ld. Performance in MFLOPS.\n",
            maximumDatasetSize, totalNumberProcessedPoints
    );


    printf("| %10s | %8s | %8s | %8s | %8s | %4s | %10s |\n", "Data size", "Teams", "Threads", "MFLOPS", "Cycles", "GPU", "Checksum");

    long datasetSize = 64;
    while (datasetSize <= maximumDatasetSize) {

        // Keep the total number of processed points constant by 
        // adjusting the number of repetitions according to data set size
        long cycles = std::clamp(totalNumberProcessedPoints / datasetSize, 8l, 65536l);

        // Run benchmark
        triad(datasetSize, cycles, numTeams, numThreads);

        datasetSize *= 2;
    }

    return 0;
}

