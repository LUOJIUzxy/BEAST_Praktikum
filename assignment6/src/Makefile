CXX=c++
CXX_FLAGS=-std=c++17 -Wall -Wextra -O3 -march=native -g -DASSIGNMENT_SOLUTION

all: vector list

vector: assignment6_part_1.c
	${CXX} ${CXX_FLAGS} -fopenmp $< -o assignment6_part_1

list: assignment6_part_2.c
	${CXX} ${CXX_FLAGS} -fopenmp $< -o assignment6_part_2

graphics_vector:
	python3 util/generate_chart_part1.py --performance-data perf_data1.txt --reference-data util/reference_output_parallel_1.txt --output-file perf_data1.svg
	python3 util/generate_chart_part1.py --performance-data perf_data1.txt --reference-data util/reference_output_parallel_1.txt --output-file perf_data1.pdf

graphics_list:
	python3 util/generate_chart_part2.py --performance-data perf_data2.txt --reference-data util/reference_output_parallel_2.txt --output-file perf_data2.svg
	python3 util/generate_chart_part2.py --performance-data perf_data2.txt --reference-data util/reference_output_parallel_2.txt --output-file perf_data2.pdf

clean:
	rm -f assignment6_part_1 assignment6_part_2 perf_data1.txt perf_data1.pdf perf_data1.svg perf_data2.txt perf_data2.pdf perf_data2.svg
