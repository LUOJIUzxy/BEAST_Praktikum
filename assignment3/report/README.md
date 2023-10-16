# Assignment 6 Report - Contributors: Oliver Yat-Sing Fung 

## 1 - Linux Perf 
In the following we are going to run 

    $perf list

on various systems and analyse the available events. 

Interesting for the roofline analysis are the following events:
* LLC_load-misses 
* L1-dcache-load-misses
* cpu-cycles
* LLC-loads
* L1-dcache-loads

These events are interesting for observing the performance bound by the bandwidth.



## 2 - Likwid  
### A - Marker API Implementation

### B - Performance Groups 
 
We can find the different performance groups with:

    $likwid-perfctr -a 

        Group name		Description
    --------------------------------------------------------------------------------
	     DATA	Load to store ratio
	   BRANCH	Branch prediction miss rate/ratio
	      MEM	Main memory bandwidth in MBytes/s
	       L2	L2 cache bandwidth in MBytes/s
	     SPEC	Information about speculative execution
	 TLB_DATA	L1 data TLB miss rate/ratio
	TLB_INSTR	L1 Instruction TLB miss rate/ratio
	   ICACHE	Instruction cache miss rate/ratio
	       L3	L3 cache bandwidth in MBytes/s
	  L2CACHE	L2 cache miss rate/ratio
	 FLOPS_DP	Double Precision MFLOP/s
	 FLOPS_SP	Single Precision MFLOP/s

* the group for the floating-point performance is - FLOPS_DP 
* the group for the memory/cache bandwidth utilization is  - MEM 


Performance metric formulas and performance events used in the ice1 system:

    $likwid-perfctr -g -MEM -H 

    Group MEM:
    Formulas:
    Memory read bandwidth [MBytes/s] = 1.0E-06*(SUM(CAS_COUNT_RD))*64.0/runtime
    Memory read data volume [GBytes] = 1.0E-09*(SUM(CAS_COUNT_RD))*64.0
    Memory write bandwidth [MBytes/s] = 1.0E-06*(SUM(CAS_COUNT_WR))*64.0/runtime
    Memory write data volume [GBytes] = 1.0E-09*(SUM(CAS_COUNT_WR))*64.0
    Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(CAS_COUNT_RD)+SUM(CAS_COUNT_WR))*64.0/runtime
    Memory data volume [GBytes] = 1.0E-09*(SUM(CAS_COUNT_RD)+SUM(CAS_COUNT_WR))*64.0

    $likwid-perfctr -g FLOPS_DP -H
    Group FLOPS_DP:
    Formulas:
    DP [MFLOP/s] = 1.0E-06*(FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE*2+FP_ARITH_INST_RETIRED_SCALAR_DOUBLE+FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE*4+FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE*8)/runtime
    AVX DP [MFLOP/s] = 1.0E-06*(FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE*4+FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE*8)/runtime
    AVX512 DP [MFLOP/s] = 1.0E-06*(FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE*8)/runtime
    Packed [MUOPS/s] = 1.0E-06*(FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE)/runtime
    Scalar [MUOPS/s] = 1.0E-06*FP_ARITH_INST_RETIRED_SCALAR_DOUBLE/runtime
    Vectorization ratio = 100*(FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE)/(FP_ARITH_INST_RETIRED_SCALAR_DOUBLE+FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE+FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE)
    -
    SSE scalar and packed double precision FLOP rates.


### C, E, F - FLOPS, Cache BW, Memory BW
### AMD A64FX system - Cray compiler
* L1D cache size 3MiB (64KiB / core) 
* L2 cache size 32 MiB (8MiB x 4)
* There is no L3 cache
* max cores 48

Cray cc : Version 10.0.1 - compiler flags:

    cc -O3 -h omp -DLIKWID_PERFMON -I/home/sw/aarch64/likwid/likwid-5.2.1-gcc-11.0.0//include -o triad triad.c -L/home/sw/aarch64/likwid/likwid-5.2.1-gcc-11.0.0//lib -llikwid

#### **Single Thread**: 
8MiB >  Data  > 64 KiB

Vector size:4096 

* MFOPS: 3481.9692 
* L2 BW:  70484.4379 MBytes/s
* MEM BW: 0.0033 MBytes/s

Data > 8MiB
Vector size: 1048576 
* MFLOPS: 3079.2357
* L2 BW: 56745.8819 MBytes/s
* MEM BW: 61751.1852  MBytes/s
#### **Multithread**:
32 MiB > Data > 3 MiB
Vector size: 262144 
* MFLOPS: 144713.8348 
* L2 BW: 3109928 MBytes/s
* MEM BW: 3202.0221 MBytes/s

Data > 32 MiB 
Vector Size 4194304: 
* MFLOPS: 48910.6763 
* L2 BW: 933843.7567 MBytes/s
* MEM BW: 899561.1416 MBytes/s

### ThunderX2 
* L1D cache 2048KB (32KB / core)
* L2 cache 16MB (256KB / core)
* L3 distributed 64MB (32MB per socket)
* 2 sockets / 32 cores = 64 cores

gcc (GCC) 8.3.1 20191121 (Red Hat 8.3.1-5) compiler flags:

    cc -O3 -fopenmp -march=native  -DLIKWID_PERFMON -I/home/sw/aarch64/likwid/likwid-5.2.1-gcc-11.0.0//include -o triad triad.c -L/home/sw/aarch64/likwid/likwid-5.2.1-gcc-11.0.0//lib -llikwid


A64FX: 

    +------------+-----------+-----------+-----------+-----------+
    |    Size    | #Threads  |   FLOPS   |   L2  BW  | Memory BW |
    +------------+-----------+-----------+-----------+-----------+
    |  >L1 size  |     1     | 3481.9692 | 70484.4379|  0.0033   |
    |  >L2 size  |     1     | 3079.2357 | 56745.8819| 61751.1852|
    | >ΣL1 size  | max. cores|144713.8348|  3109928  | 3202.0221 |
    | >ΣL2 size  | max. cores| 48910.6763|933843.7567|899561.1416|
    +------------+-----------+-----------+-----------+-----------+


## 3 - GPU Profiling

After running:

    nsys nvprof ./gpu 67108864 67108864

and 

    nsys stats report.qdrep

We get following output: 
    
     Time(%)  Total Time (ns)  Num Calls    Average      Minimum      Maximum             Name        
     -------  ---------------  ---------  ------------  ---------  -------------  --------------------
        92.2    6,967,990,105        147  47,401,293.2      2,345  2,103,442,380  cuStreamSynchronize 
         3.9      297,589,820         84   3,542,736.0      8,735     45,088,215  cuMemcpyHtoDAsync_v2
         2.7      203,253,580         42   4,839,371.0     35,840    101,587,630  cuMemcpyDtoHAsync_v2
         0.9       64,354,785        105     612,902.7      3,670     48,844,170  cuMemFree_v2        
         0.2       15,302,765        105     145,740.6      3,465        677,590  cuMemAlloc_v2       
         0.1        4,106,350          1   4,106,350.0  4,106,350      4,106,350  cuModuleLoadDataEx  
         0.0        2,303,040          1   2,303,040.0  2,303,040      2,303,040  cuModuleUnload      
         0.0        1,325,910         63      21,046.2     12,565        174,530  cuLaunchKernel      
         0.0          477,885         32      14,933.9      2,970        205,600  cuStreamCreate      
         0.0          178,905         32       5,590.8      3,420         32,625  cuStreamDestroy_v2  
         0.0          129,215          3      43,071.7     36,085         56,265  cuMemcpyDtoH_v2     
         0.0           12,305          1      12,305.0     12,305         12,305  cuMemcpyHtoD_v2     


     Time(%)  Total Time (ns)  Instances     Average      Minimum       Maximum                    Name                
     -------  ---------------  ---------  -------------  ----------  -------------  -----------------------------------
        60.5    4,206,940,755         21  200,330,512.1       8,610  2,103,389,285  __omp_offloading_32_ab0324__Z5_2...
        39.5    2,743,014,300         21  130,619,728.6  29,923,950    456,242,760  __omp_offloading_32_ab0324__Z5_3...
         0.0           98,385         21        4,685.0       4,220          6,465  __omp_offloading_32_ab0324__Z5_1...


     Time(%)  Total Time (ns)  Operations    Average    Minimum    Maximum        Operation     
     -------  ---------------  ----------  -----------  -------  -----------  ------------------
        59.8      296,050,315          85  3,482,944.9    1,405   45,024,275  [CUDA memcpy HtoD]
        40.2      198,769,495          45  4,417,099.9    1,535  101,233,010  [CUDA memcpy DtoH]


         Total      Operations   Average    Minimum    Maximum        Operation     
     -------------  ----------  ----------  -------  -----------  ------------------
     3,145,726.586          85  37,008.548    0.004  524,288.000  [CUDA memcpy HtoD]
     1,048,575.585          45  23,301.680    0.001  524,288.000  [CUDA memcpy DtoH]


Unfortunately I couldn't get ncu to work, due to the following error: 

running:

     ncu --metrics=launch__thread_count,launch__grid_size,launch__block_size --target-processes=all ./gpu 67108864 67108864

results in this error: 
    ==ERROR== Failed to connect to process 2911851.

### B - Rocprof on rome2:

I unfortunately also run into some problems here: 

After loading any rocprof module I got following error after running: 

    rocprof --list-basic

    RPL: on '221117_143421' from '/home/di75gix/spack_zen/opt/spack/linux-sles15-zen2/gcc-9.3.1/rocprofiler-dev-4.5.2-tekcnzfgxv32goopgrnxu3jx75xmkvht/rocprofiler' in '/home/h039y12'
    Basic HW counters:
    aqlprofile API table load failed: HSA_STATUS_ERROR: A generic error has occurred.
    /home/di75gix/spack_zen/opt/spack/linux-sles15-zen2/gcc-9.3.1/rocprofiler-dev-4.5.2-tekcnzfgxv32goopgrnxu3jx75xmkvht/bin/rocprof: line 373: 212293 Aborted                 (core dumped) /home/di75gix/spack_zen/opt/spack/linux-sles15-zen2/gcc-9.3.1/rocprofiler-dev-4.5.2-tekcnzfgxv32goopgrnxu3jx75xmkvht/rocprofiler/tool/ctrl

### GPU Tracing - THAPI:



























