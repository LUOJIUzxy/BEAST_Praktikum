LIKWID_BASE ?= $(shell dirname $(dir $(shell which likwid-perfctr)))
CFLAGS += -O3 -fopenmp -march=native  -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
CFLAGS_CPU += -O3 -fopenmp   -march=native 
CFLAGS_AFX += -O3 -h omp    -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
CFLAGS_AFX_OLD += -O3 -h omp  
LDFLAGS := -L$(LIKWID_BASE)/lib -llikwid

all: triad

triad_CS: 
	$(CC) $(CFLAGS_AFX) -o triad triad.c $(LDFLAGS)
triad:
	$(CC) $(CFLAGS) -o triad triad.c $(LDFLAGS)

triad_cpu:
		$(CC) $(CFLAGS_CPU) -o triad_cpu triad_cpu.c

clean_cpu:
	rm -f triad_cpu
clean:
	rm -f triad


#//likwid-perfctr -g FLOPS_DP  -m ./triad 67108864 67108864
