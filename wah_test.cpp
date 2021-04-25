#include "wah_test.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iterator>
#define MISMATCH_MAX 20


void Test(UINT* (*tested_function)(UINT*), UINT* data, UINT* expected, std::string test_name)
{
	printf("TEST '%s' - Input Size: %u \n", test_name.c_str(), sizeof(data) / sizeof(UINT));
	UINT* actual = tested_function(data);
	if (actual == nullptr)
	{
		printf("FAILED: function did not return any data!");
		return;
	}
	if (sizeof(actual) != sizeof(expected))
	{
		printf("FAILED: Tables' sizes mismatch! Expected: %d, Actual: %d\n"
		, sizeof(expected)/sizeof(UINT)
		, sizeof(actual)/sizeof(UINT));
		return;
	}
	int size = (sizeof(actual)) / sizeof(UINT);
	int mismatches = 0;
	for (int i = 0; i < size; i++)
	{
		if (actual[i] != expected[i])
		{
			mismatches++;
			if (mismatches<=MISMATCH_MAX)
			{
				printf("Tables' values mismatch on position %d! Expected: %d, Actual: %d\n"
					, i
					, expected[i]
					, actual[i]);
				return;
			}
		}
	}
	if (mismatches == 0) printf("PASSED\n");
	else printf("FAILED: Found mismatches. Count: %d\n", mismatches);
}


void Benchmark(UINT* (*benchmark_function)(UINT*), UINT* data, int repeats, std::string bench_name, bool print_times)
{
	printf("BENCHMARK '%s' - Input Size: %u \n", bench_name.c_str(), sizeof(data) / sizeof(UINT));
	cudaEvent_t start, stop;
	float max = -1;
	float mean = 0;
	float min = 0;
	for (int i = 0; i < repeats; i++)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		benchmark_function(data);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float exec_time = 0;
		cudaEventElapsedTime(&exec_time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		exec_time /= 1000;
		if(print_times) printf("RUN %d: %f\n", i+1 , exec_time );
		if (max < exec_time) max = exec_time;
		if (i == 0 || min > exec_time) min = exec_time;
		mean += exec_time;
	}
	printf("MIN: %fs | MEAN: %fs | MAX: %fs \n", min, mean / repeats, max);
}

