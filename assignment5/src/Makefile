ifeq ($(NODE),thx) # ThunderX systems
	#@module load cuda/11.1.1
	#module load llvm/11.0.0_nvptx_offloading
	CXX = clang++
	CXX_FLAGS = -std=c++17 -Wall -Wextra -Ofast -g -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70
else # Rome systems
	CXX = /opt/rocm/llvm/bin/clang++
	CXX_FLAGS = -std=c++17 -Wall -Wextra -Ofast -g -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
endif

all: assignment5

assignment5: assignment5.cpp
	${CXX} ${CXX_FLAGS} assignment5.cpp -o assignment5


# For this assignment we don't have CI tests.

clean:
	rm -f assignment5 util/part2/perf_data.txt util/perf_data.pdf util/perf_data.svg util/perf_data.txt util/perf_data.pdf util/perf_data.svg
