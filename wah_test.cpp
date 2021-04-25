#include "wah_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define MISMATCH_MAX 20

UINT* CpuWAH(UINT* data)
{

}

void Test(UINT* (*tested_function)(int,UINT*),int data_size, UINT* data, int expected_size, UINT* expected, std::string test_name)
{
	int dsize = data_size;
	UINT* d_data;
	cudaMalloc((UINT**)&d_data, sizeof(UINT) * dsize);
	cudaMemcpy(d_data, data, sizeof(UINT) * dsize, cudaMemcpyHostToDevice);

	printf("TEST '%s' - Input Size: %u \n", test_name.c_str(), dsize);
	printf("=======================================================\n");
	UINT* actual = tested_function(dsize,d_data);
	printf("=======================================================\n");
	cudaFree(d_data);
	if (actual == nullptr)
	{
		printf("FAILED: function did not return any data!");
		return;
	}
	int size = expected_size;
	int mismatches = 0;
	for (int i = 0; i < size; i++)
	{
		if (actual[i] != expected[i])
		{
			mismatches++;
			if (mismatches<=MISMATCH_MAX)
			{
				printf("Tables' values mismatch on position %d! Expected: %u, Actual: %u\n"
					, i
					, expected[i]
					, actual[i]);
			}
		}
	}
	delete[] actual;
	if (mismatches == 0) printf("PASSED\n");
	else printf("FAILED: Found mismatches. Count: %d\n", mismatches);
	
}


void Benchmark(UINT* (*benchmark_function)(int,UINT*), int data_size, UINT* d_data, int repeats, std::string bench_name, bool print_times)
{
	int size = data_size;


	printf("BENCHMARK '%s' - Input Size: %u \n", bench_name.c_str(), size);
	cudaEvent_t start, stop;
	float max = -1;
	float mean = 0;
	float min = 0;
	for (int i = 0; i < repeats; i++)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		if(size<=64) printf("=======================================================\n");
		UINT * result = benchmark_function(size,d_data);
		if(size<=64) printf("=======================================================\n");
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
		delete[] result;
	}
	printf("MIN: %fs | MEAN: %fs | MAX: %fs \n", min, mean / repeats, max);
}



void NoCompressTest(UINT* (*tested_function)(int, UINT*))
{
	int size = 32;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 127;
	}
	Test(tested_function, size, table, size, table, "Not compressable Input Test");
	delete[] table;
}

void AllCompressTest(UINT* (*tested_function)(int, UINT*))
{
	int size = 32;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 0;
	}
	UINT* result = new UINT[1];
	result[0] = 0x80000020;
	Test(tested_function, size, table, 1, result, "All Zeros Test");
	delete[] table;
	delete[] result;
}

void UnitTests(UINT* (*tested_function)(int, UINT*))
{
	printf("Preforming unit tests...\n");
	NoCompressTest(tested_function);
	AllCompressTest(tested_function);
}