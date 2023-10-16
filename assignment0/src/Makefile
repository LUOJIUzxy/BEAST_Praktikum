#CFLAGS=-O0 -march=native
#CFLAGS=-O1 -march=native
#CFLAGS=-O2 -march=native
#CFLAGS=-O3 -march=native
#
# Refer: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

# CC is the environment variable for the C-Compiler
#
# use either CC=gcc on any of the systems or:
#
# cs2:   cc (Cray C-Compiler)
# ice2:  icc   / icpc  / clang
# thx2:  clang / cc [need to cross-compile on cs2]
# rome2: aomp  / clang
#
# example to compile with gcc and run branch_cond:
#
# CC=gcc CFLAGS="-O3 -march=native" make && make run

LLIKWID = -llikwid

all: branch_cond

branch_cond:
	$(CC) $(CFLAGS) -o branch_cond branch_cond.c
	
# with_likwid:
# 	$(CC) $(CFLAGS) -o branch_cond branch_cond.c -DLIKWID_PERFMON $(LLIKWID)
	
run:
	./branch_cond

# in case, you want to use likwid to measure branch prediction rate/ratio
#	likwid-perfctr -C S0:0 -g BRANCH ./branch_cond

clean:
	rm -f branch_cond
