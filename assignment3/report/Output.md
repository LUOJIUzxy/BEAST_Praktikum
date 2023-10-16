8MiB >  Data  > 64 KiB
4096 > L1 size: 

Single Thread:

FLOPS_DP

    +--------------------------+--------------+
    |          Metric          |  HWThread 0  |
    +--------------------------+--------------+
    |    Runtime (RDTSC) [s]   |       0.6167 |
    |        Clock [MHz]       |    1798.3134 |
    |            CPI           |       1.0272 |
    |     DP (FP) [MFLOP/s]    | 1.945702e-05 |
    | DP (FP+SVE128) [MFLOP/s] |     870.4923 |
    | DP (FP+SVE256) [MFLOP/s] |    1740.9846 |
    | DP (FP+SVE512) [MFLOP/s] |    3481.9692 |
    +--------------------------+--------------+

L2

    +------------------------------------+------------+
    |               Metric               | HWThread 0 |
    +------------------------------------+------------+
    |         Runtime (RDTSC) [s]        |     0.6236 |
    |                 CPI                |     1.0385 |
    |  L1D<-L2 load bandwidth [MBytes/s] | 55979.1828 |
    |  L1D<-L2 load data volume [GBytes] |    34.9107 |
    | L1D->L2 evict bandwidth [MBytes/s] | 14505.1533 |
    | L1D->L2 evict data volume [GBytes] |     9.0460 |
    |  L1I<-L2 load bandwidth [MBytes/s] |     0.1018 |
    |  L1I<-L2 load data volume [GBytes] |     0.0001 |
    |    L1<->L2 bandwidth [MBytes/s]    | 70484.4379 |
    |    L1<->L2 data volume [GBytes]    |    43.9568 |
    +------------------------------------+------------+

MEM

    +-----------------------------------+--------------+
    |               Metric              |  HWThread 0  |
    +-----------------------------------+--------------+
    |        Runtime (RDTSC) [s]        |       0.6134 |
    |                CPI                |       1.0216 |
    |  Memory read bandwidth [MBytes/s] |       0.0029 |
    |  Memory read data volume [GBytes] | 1.792000e-06 |
    | Memory write bandwidth [MBytes/s] |       0.0004 |
    | Memory write data volume [GBytes] | 2.560000e-07 |
    |    Memory bandwidth [MBytes/s]    |       0.0033 |
    |    Memory data volume [GBytes]    | 2.048000e-06 |
    +-----------------------------------+--------------+


1048576 > L2 size: 

FLOPS_DP

    +--------------------------+--------------+
    |          Metric          |  HWThread 0  |
    +--------------------------+--------------+
    |    Runtime (RDTSC) [s]   |       0.6974 |
    |        Clock [MHz]       |    1797.2621 |
    |            CPI           |       1.1673 |
    |     DP (FP) [MFLOP/s]    | 1.720657e-05 |
    | DP (FP+SVE128) [MFLOP/s] |     769.8089 |
    | DP (FP+SVE256) [MFLOP/s] |    1539.6179 |
    | DP (FP+SVE512) [MFLOP/s] |    3079.2357 |
    +--------------------------+--------------+


L2

    +------------------------------------+------------+
    |               Metric               | HWThread 0 |
    +------------------------------------+------------+
    |         Runtime (RDTSC) [s]        |     0.7645 |
    |                 CPI                |     1.2791 |
    |  L1D<-L2 load bandwidth [MBytes/s] | 45345.6322 |
    |  L1D<-L2 load data volume [GBytes] |    34.6654 |
    | L1D->L2 evict bandwidth [MBytes/s] | 11400.1677 |
    | L1D->L2 evict data volume [GBytes] |     8.7151 |
    |  L1I<-L2 load bandwidth [MBytes/s] |     0.0820 |
    |  L1I<-L2 load data volume [GBytes] |     0.0001 |
    |    L1<->L2 bandwidth [MBytes/s]    | 56745.8819 |
    |    L1<->L2 data volume [GBytes]    |    43.3806 |
    +------------------------------------+------------+

MEM

    +-----------------------------------+------------+
    |               Metric              | HWThread 0 |
    +-----------------------------------+------------+
    |        Runtime (RDTSC) [s]        |     0.6956 |
    |                CPI                |     1.1642 |
    |  Memory read bandwidth [MBytes/s] | 49400.1856 |
    |  Memory read data volume [GBytes] |    34.3608 |
    | Memory write bandwidth [MBytes/s] | 12350.9996 |
    | Memory write data volume [GBytes] |     8.5909 |
    |    Memory bandwidth [MBytes/s]    | 61751.1852 |
    |    Memory data volume [GBytes]    |    42.9517 |
    +-----------------------------------+------------+

Thread 48 


32 MiB > Data > 3 MiB
262144 > L1 Cache

FLOPS_DP

    +-------------------------------+-------------+-----------+-----------+-----------+
    |             Metric            |     Sum     |    Min    |    Max    |    Avg    |
    +-------------------------------+-------------+-----------+-----------+-----------+
    |    Runtime (RDTSC) [s] STAT   |      0.7130 |    0.0143 |    0.0154 |    0.0149 |
    |        Clock [MHz] STAT       |  85510.6163 | 1778.3141 | 1783.1003 | 1781.4712 |
    |            CPI STAT           |     56.2551 |    1.1297 |    1.2157 |    1.1720 |
    |     DP (FP) [MFLOP/s] STAT    |      0.0384 |    0.0008 |    0.0008 |    0.0008 |
    | DP (FP+SVE128) [MFLOP/s] STAT |  36178.4877 |  725.6051 |  782.2637 |  753.7185 |
    | DP (FP+SVE256) [MFLOP/s] STAT |  72356.9371 | 1451.2094 | 1564.5265 | 1507.4362 |
    | DP (FP+SVE512) [MFLOP/s] STAT | 144713.8348 | 2902.4181 | 3129.0522 | 3014.8716 |
    +-------------------------------+-------------+-----------+-----------+-----------+

L2

    +-----------------------------------------+--------------+--------------+------------+------------+
    |                  Metric                 |      Sum     |      Min     |     Max    |     Avg    |
    +-----------------------------------------+--------------+--------------+------------+------------+
    |         Runtime (RDTSC) [s] STAT        |       0.7027 |       0.0135 |     0.0155 |     0.0146 |
    |                 CPI STAT                |      55.4544 |       1.0658 |     1.2255 |     1.1553 |
    |  L1D<-L2 load bandwidth [MBytes/s] STAT | 2.431814e+06 |   47849.9833 | 54029.3270 | 50662.7937 |
    |  L1D<-L2 load data volume [GBytes] STAT |      35.5868 |       0.7289 |     0.7515 |     0.7414 |
    | L1D->L2 evict bandwidth [MBytes/s] STAT |  677956.8737 |   13086.7803 | 15561.5540 | 14124.1015 |
    | L1D->L2 evict data volume [GBytes] STAT |       9.9235 |       0.1933 |     0.2277 |     0.2067 |
    |  L1I<-L2 load bandwidth [MBytes/s] STAT |     156.9339 |       2.9443 |     3.6777 |     3.2695 |
    |  L1I<-L2 load data volume [GBytes] STAT |       0.0027 | 4.403200e-05 |     0.0001 |     0.0001 |
    |    L1<->L2 bandwidth [MBytes/s] STAT    | 3.109928e+06 |   60975.5574 | 68726.3886 | 64790.1647 |
    |    L1<->L2 data volume [GBytes] STAT    |      45.5134 |       0.9223 |     0.9783 |     0.9482 |
    +-----------------------------------------+--------------+--------------+------------+------------+
    
MEM 

    +----------------------------------------+-----------+--------------+--------------+--------------+
    |                 Metric                 |    Sum    |      Min     |      Max     |      Avg     |
    +----------------------------------------+-----------+--------------+--------------+--------------+
    |        Runtime (RDTSC) [s] STAT        |    0.7040 |       0.0137 |       0.0153 |       0.0147 |
    |                CPI STAT                |   55.5134 |       1.0838 |       1.2037 |       1.1565 |
    |  Memory read bandwidth [MBytes/s] STAT | 3152.0121 |       3.1432 |     455.4609 |      65.6669 |
    |  Memory read data volume [GBytes] STAT |    0.0468 | 4.710400e-05 |       0.0063 |       0.0010 |
    | Memory write bandwidth [MBytes/s] STAT |   50.0100 |       0.4100 |       2.1726 |       1.0419 |
    | Memory write data volume [GBytes] STAT |    0.0007 | 6.144000e-06 | 3.148800e-05 | 1.524800e-05 |
    |    Memory bandwidth [MBytes/s] STAT    | 3202.0221 |       3.5532 |     456.8395 |      66.7088 |
    |    Memory data volume [GBytes] STAT    |    0.0474 |       0.0001 |       0.0063 |       0.0010 |
    +----------------------------------------+-----------+--------------+--------------+--------------+


Data > 32 MiB 
4194304 > L2 Cache 

FLOPS_DP

    +-------------------------------+------------+-----------+-----------+-----------+
    |             Metric            |     Sum    |    Min    |    Max    |    Avg    |
    +-------------------------------+------------+-----------+-----------+-----------+
    |    Runtime (RDTSC) [s] STAT   |     2.1305 |    0.0263 |    0.0500 |    0.0444 |
    |        Clock [MHz] STAT       | 85448.2127 | 1774.0002 | 1785.5960 | 1780.1711 |
    |            CPI STAT           |   169.3923 |    2.0940 |    3.9688 |    3.5290 |
    |     DP (FP) [MFLOP/s] STAT    |     0.0137 |    0.0002 |    0.0005 |    0.0003 |
    | DP (FP+SVE128) [MFLOP/s] STAT | 12227.6785 |  223.5021 |  425.9716 |  254.7433 |
    | DP (FP+SVE256) [MFLOP/s] STAT | 24455.3444 |  447.0039 |  851.9427 |  509.4863 |
    | DP (FP+SVE512) [MFLOP/s] STAT | 48910.6763 |  894.0076 | 1703.8850 | 1018.9724 |
    +-------------------------------+------------+-----------+-----------+-----------+

L2

    +-----------------------------------------+-------------+--------------+------------+------------+
    |                  Metric                 |     Sum     |      Min     |     Max    |     Avg    |
    +-----------------------------------------+-------------+--------------+------------+------------+
    |         Runtime (RDTSC) [s] STAT        |      2.2465 |       0.0421 |     0.0499 |     0.0468 |
    |                 CPI STAT                |    178.9790 |       3.3557 |     3.9769 |     3.7287 |
    |  L1D<-L2 load bandwidth [MBytes/s] STAT | 741758.1941 |   14435.9486 | 17185.4813 | 15453.2957 |
    |  L1D<-L2 load data volume [GBytes] STAT |     34.6684 |       0.7202 |     0.7236 |     0.7223 |
    | L1D->L2 evict bandwidth [MBytes/s] STAT | 192035.6553 |    3734.0177 |  4407.5201 |  4000.7428 |
    | L1D->L2 evict data volume [GBytes] STAT |      8.9756 |       0.1852 |     0.1887 |     0.1870 |
    |  L1I<-L2 load bandwidth [MBytes/s] STAT |     49.9072 |       0.9294 |     1.2761 |     1.0397 |
    |  L1I<-L2 load data volume [GBytes] STAT |      0.0026 | 4.608000e-05 |     0.0001 |     0.0001 |
    |    L1<->L2 bandwidth [MBytes/s] STAT    | 933843.7567 |   18170.9095 | 21594.2369 | 19455.0783 |
    |    L1<->L2 data volume [GBytes] STAT    |     43.6462 |       0.9068 |     0.9114 |     0.9093 |
    +-----------------------------------------+-------------+--------------+------------+------------+

MEM 

    +----------------------------------------+-------------+------------+------------+------------+
    |                 Metric                 |     Sum     |     Min    |     Max    |     Avg    |
    +----------------------------------------+-------------+------------+------------+------------+
    |        Runtime (RDTSC) [s] STAT        |      2.2476 |     0.0264 |     0.0521 |     0.0468 |
    |                CPI STAT                |    178.7804 |     2.1087 |     4.1434 |     3.7246 |
    |  Memory read bandwidth [MBytes/s] STAT | 720516.6075 | 12845.3505 | 16393.0148 | 15010.7627 |
    |  Memory read data volume [GBytes] STAT |     33.7074 |     0.3390 |     0.7250 |     0.7022 |
    | Memory write bandwidth [MBytes/s] STAT | 179044.5340 |  2898.9114 |  4218.5815 |  3730.0945 |
    | Memory write data volume [GBytes] STAT |      8.3811 |     0.0765 |     0.1942 |     0.1746 |
    |    Memory bandwidth [MBytes/s] STAT    | 899561.1416 | 15744.2619 | 20496.9646 | 18740.8571 |
    |    Memory data volume [GBytes] STAT    |     42.0882 |     0.4155 |     0.9099 |     0.8768 |
    +----------------------------------------+-------------+------------+------------+------------+



ThunderX2
Single Thread:
Vector size: 16384
FLOPS_DP

    +---------------------+--------------+
    |        Metric       |  HWThread 0  |
    +---------------------+--------------+
    | Runtime (RDTSC) [s] |       1.1272 |
    |     Clock [MHz]     |    2504.4455 |
    |         CPI         |       0.6572 |
    |     DP [MFLOP/s]    |    1578.5498 |
    |  NEON DP [MFLOP/s]  |    1578.5497 |
    |   Packed [MUOPS/s]  |     789.2749 |
    |   Scalar [MUOPS/s]  | 4.879338e-05 |
    | Vectorization ratio |     100.0000 |
    +---------------------+--------------+

L2 BW:

    +--------------------------------+------------+
    |             Metric             | HWThread 0 |
    +--------------------------------+------------+
    |       Runtime (RDTSC) [s]      |     1.1202 |
    |           Clock [MHz]          |  2503.9300 |
    |               CPI              |     0.6530 |
    |  L2D load bandwidth [MBytes/s] | 30672.3158 |
    |  L2D load data volume [GBytes] |    34.3598 |
    | L2D evict bandwidth [MBytes/s] |  7670.1414 |
    | L2D evict data volume [GBytes] |     8.5923 |
    |  L2I load bandwidth [MBytes/s] |     0.0745 |
    |  L2I load data volume [GBytes] |     0.0001 |
    |     L2 bandwidth [MBytes/s]    | 38342.5317 |
    |     L2 data volume [GBytes]    |    42.9522 |
    +--------------------------------+------------+

