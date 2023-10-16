#ifndef _CUTIL_H_
#define _CUTIL_H_
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        std::cerr<<"Cuda error "<< cudaGetErrorString(err) ; \
        exit(EXIT_FAILURE);                                                  \
    } }


using namespace std;

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))
__global__ void MM(float* a,
		float* b, float* c, 
		int N, unsigned long repetitions ) ;

__global__ void sharedTiledMM(double*a, double* b, double*c, int N);


// Task f: Some PTX code :) What is PTX ? Why is this here ? Why will this cause problems ? :) 
__device__ __forceinline__ unsigned long long __globaltimer() {
	unsigned long long globaltimer;
	asm volatile ("mov.u64 %0, %globaltimer;"   : "=l"(globaltimer));
	return globaltimer;
}



double cpu_matrix_mult_checksum(double *h_a, double *h_b, 
		int n,  int REP) {
    double sum = 0 ;

for(int r=0; r<REP; ++r){
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            double tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * n + j];
            }
            //h_result[i * n + j] = tmp;
	    sum+=tmp;
        }
    }
 }
    return sum;
}

void Checksum ( int N, double* vector,  double expected_result) {

    double checksum = 0;
    for (int i = 0; i < N; i++) {
	    for (int j= 0; j < N ; ++j){
        	checksum += vector[i*N + j];
	    } 
    }

    if(abs(checksum - expected_result) > 0.001){
    	printf("Wrong result for M=%4d. expected %.3f but got %.3f. Aborting...\n", N, expected_result, checksum);
    	exit(EXIT_FAILURE);
    }
    return;
}

static inline int _ConvertSMVer2Cores(int major, int minor){
	switch(major){
		case 1:  return 8;
		case 2:  switch(minor){
			case 1:  return 48;
			default: return 32;
		}
		case 3:  return 192;
		case 6: switch(minor){
			case 0:  return 64;
			default: return 128;
		}
		case 7:  return 64;
		default: return 128;
	}
}

static inline bool IsFP16Supported(void){
	cudaDeviceProp deviceProp;
	int current_device;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	return deviceProp.major>5 || (deviceProp.major == 5 && deviceProp.minor == 3);
}

static inline void GetDevicePeakInfo(double *aGIPS, double *aGBPS, cudaDeviceProp *aDeviceProp = NULL){
	cudaDeviceProp deviceProp;
	int current_device;
	if( aDeviceProp )
		deviceProp = *aDeviceProp;
	else{
		CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	}
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	*aGIPS = 1000.0 * deviceProp.clockRate * TotalSPs / (1000.0 * 1000.0 * 1000.0);  // Giga instructions/sec
	*aGBPS = 2.0 * (double)deviceProp.memoryClockRate * 1000.0 * (double)deviceProp.memoryBusWidth / 8.0;
}

static inline cudaDeviceProp GetDeviceProperties(void){
	cudaDeviceProp deviceProp;
	int current_device;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	return deviceProp;
}

// Print basic device information
static void PrintDeviceInfo(void){
	cudaDeviceProp deviceProp;
	int current_device, driver_version;
	CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
	CUDA_SAFE_CALL( cudaDriverGetVersion(&driver_version) );
	std::cout << left<<setw(32)<< "------------------------ Device specifications ------------------------"<<std::endl;
	std::cout << left<<setw(32)<< "Device:"<< setw(40)<<		left<< deviceProp.name<<std::endl;
	std::cout << left<<setw(32)<< "CUDA driver version:"<<		left<< setw(2)<< driver_version/1000 << left<< setw(1)<<"."<<left << setw(37) << driver_version%1000<<std::endl;
	std::cout << left<<setw(32)<< "GPU clock rate:"<<		left<< setw(4)<<"MHz "<<left<<setw(36)<< deviceProp.clockRate/1000<<std::endl;
	std::cout << left<<setw(32)<< "Memory clock rate:"<<		left<< setw(4)<<"MHz "<<left<<setw(36)<< deviceProp.memoryClockRate/1000/2<<std::endl;
	std::cout << left<<setw(32)<< "Memory bus width:"<<  		left<< setw(5)<<left << left<<setw(25)<<setw(5)<< deviceProp.memoryBusWidth<<std::endl;
	std::cout << left<<setw(32)<< "WarpSize:"<<  			left<< setw(40)<< deviceProp.warpSize<<std::endl;
	std::cout << left<<setw(32)<< "L2 cache size:"<<		left<< setw(3)<<"KB "<<left <<setw(37)<< deviceProp.l2CacheSize/1024<<std::endl;
	std::cout << left<<setw(32)<< "Total global mem:"<<		left<< setw(6)<< deviceProp.totalGlobalMem/(1024*1024)<<" MB"<<std::endl;
	std::cout << left<<setw(32)<< "Total shared mem. per Block:"<<	left<< setw(6)<< deviceProp.sharedMemPerBlock/(1024*1024)<<" MB"<<std::endl;
	std::cout << left<<setw(32)<< "ECC enabled:"<<  		left<< setw(40)<< (deviceProp.ECCEnabled?"Yes":"No") <<std::endl;
	std::cout << left<<setw(32)<< "Compute Capability:"<< 		left<< setw(1)<< deviceProp.major << left << setw(1) << "."<<left<<setw(38) << deviceProp.minor <<std::endl;
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
	std::cout << left<<setw(32)<< "Total SPs:"<< 			left<< setw(5)<< TotalSPs<<"( "<< deviceProp.multiProcessorCount<<"MPs x "<< _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)<<" SPs/MP"<<std::endl;
	double InstrThroughput, MemBandwidth;
	GetDevicePeakInfo(&InstrThroughput, &MemBandwidth, &deviceProp);
	std::cout << left<<setw(32)<< "Compute throughput:"<<  		left<< setw(7)<< InstrThroughput<<"GFlops (theoretical double precision FMAs)"<<std::endl;
	std::cout << left<<setw(32)<< "Memory bandwidth:"<<  		left << setw(8)<< MemBandwidth/(1000.0*1000.0*1000.0)<<" GB/sec"<<std::endl;
	std::cout << left<<setw(32)<< "-----------------------------------------------------------------------"<<std::endl;
}
 #endif
