#include "wah_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bit_functions.cuh"
#include "defines.h"
#include <vector>
#include <fstream>
#include <string>
#define MISMATCH_MAX 20

int last_wah_count = 0;

UINT* CpuWAH(int data_size,UINT* data)
{
	std::vector<UINT> result;
	int which = -1;
	int sum = 0;
	for (int i = 0; i < data_size; i++)
	{
		if (sum == COMPRESS_MAX)
		{
			result.push_back(get_compressed(sum, which));
			sum = 0;
			which = -1;
		}
		bool zeros = is_zeros(data[i]);
		bool ones = is_ones(data[i]);
		if (!ones && !zeros)
		{
			if (sum > 0)
			{
				result.push_back(get_compressed(sum, which));
				sum = 0;
				which = -1;
			}
			result.push_back(data[i]);
		}
		else if (zeros)
		{
			if (which == -1)
			{
				sum = 1;
				which = 0;
			}
			else if (which == 1)
			{
				result.push_back(get_compressed(sum, which));
				sum = 1;
				which = 0;
			}
			else sum++;
		}
		else
		{
			if (which == -1)
			{
				sum = 1;
				which = 1;
			}
			else if (which == 0)
			{
				result.push_back(get_compressed(sum, which));
				sum = 1;
				which = 1;
			}
			else sum++;
		}
	}
	if (sum > 0)
	{
		result.push_back(get_compressed(sum, which));
	}
	UINT* ret = new UINT[result.size()];
	for (int i = 0; i < result.size(); i++)
	{
		ret[i] = result[i];
	}
	last_wah_count = result.size();
	result.clear();
	return ret;
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
				printf("Tables' values mismatch on position %d!\n");
				printf("\tExpected: "); printBits(sizeof(UINT), &expected[i]);
				printf("\tActual:   "); printBits(sizeof(UINT), &actual[i]);
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

	std::ofstream log("results_"+std::to_string(GPU_THREADS_COUNT)+".csv", std::ios_base::app | std::ios_base::out);
	//printf("BENCHMARK '%s' - Input Size: %u \n", bench_name.c_str(), size);

	cudaEvent_t start, stop;
	float max = -1;
	float mean = 0;
	float min = 0;
	for (int i = 0; i < repeats; i++)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		//if(size<=64) printf("=======================================================\n");
		UINT * result = benchmark_function(size,d_data);
		//if(size<=64) printf("=======================================================\n");
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float exec_time = 0;
		cudaEventElapsedTime(&exec_time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		exec_time /= 1000;
		if(print_times) printf("RUN %d: %f\n", i+1 , exec_time );
		std::string resultRow;
		resultRow += bench_name;
		resultRow += ";";
		resultRow += std::to_string(data_size*32);
		resultRow += ";";
		resultRow += std::to_string(exec_time);
		resultRow += ";";
		log << resultRow << std::endl;
		if (max < exec_time) max = exec_time;
		if (i == 0 || min > exec_time) min = exec_time;
		mean += exec_time;
		delete[] result;
	}
	log.close();
	//printf("MIN: %fs | MEAN: %fs | MAX: %fs \n", min, mean / repeats, max);
}



void NoCompressTest(UINT* (*tested_function)(int, UINT*))
{
	int size = 64;
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

void BetweenWarpsMerge(UINT* (*tested_function)(int, UINT*))
{
	int size = 64;
	UINT* table = new UINT[size];
	for (int i = 0; i < 35; i++)
	{
		table[i] = 0;
	}
	for (int i = 35; i < size; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	UINT* result = new UINT[2];
	result[0] = get_compressed(35,0);
	result[1] = get_compressed(29,1);
	Test(tested_function, size, table, 2, result, "Merge Between Warps Test");
	delete[] table;
	delete[] result;
}

void MixedCompressions(UINT* (*tested_function)(int, UINT*))
{
	int size = 64;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		if(i%8<=3) table[i] = 0;
		else table[i] = 0x7FFFFFFF;
	}
	UINT* result = new UINT[16];
	for (int i = 0; i < 16; i++)
	{
		result[i] = get_compressed(4, i%2);
	}
	Test(tested_function, size, table, 16, result, "Mixed Compressions Test");
	delete[] table;
	delete[] result;
}

void MixedCompressionWithLiterals(UINT* (*tested_function)(int, UINT*))
{
	int size = 32;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 128;
	}
	table[13] = 0;
	table[14] = 0;
	table[15] = 0;
	table[16] = 0;
	
	table[22] = 0x7FFFFFFF;
	table[23] = 0x7FFFFFFF;

	table[29] = 0x7FFFFFFF;
	UINT* result = new UINT[28];
	for (int i = 0; i < 28; i++)
	{
		result[i] = 128;
	}
	result[13] = get_compressed(4, 0);
	result[19] = get_compressed(2, 1);
	result[25] = get_compressed(1, 1);
	Test(tested_function, size, table, 28, result, "Mixed Compress with Literals");
	delete[] table;
	delete[] result;
}

void AllVariants_1(UINT* (*tested_function)(int, UINT*))
{
	int size = 64;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 128;
	}
	for (int i = 29; i < 34; i++)
	{
		table[i] = 0;
	}
	for (int i = 13; i < 15; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	for (int i = 16; i < 18; i++)
	{
		table[i] = 0;
	}
	table[19] = 0x7FFFFFFF;
	for (int i = 0; i < 6; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	for (int i = 49; i < 56; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	for (int i = 61; i < 64; i++)
	{
		table[i] = 0;
	}
	table[37] = 0;
	
	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Mixed Compress 1 (Every Scenario)");
	delete[] table;
	delete[] result;
}

void MultipleWarpsMerge(UINT* (*tested_function)(int, UINT*))
{
	int size = 32*10;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 0;
	}
	table[37] = 352;
	table[134] = 123;
	table[256] = 5453;

	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Multiple Warps Merge");
	delete[] table;
	delete[] result;
}

void UnitTests(UINT* (*tested_function)(int, UINT*))
{
	printf("Performing unit tests...\n");
	NoCompressTest(tested_function);
	AllCompressTest(tested_function);
	BetweenWarpsMerge(tested_function);
	MixedCompressions(tested_function);
	MixedCompressionWithLiterals(tested_function);
	AllVariants_1(tested_function);
	MultipleWarpsMerge(tested_function);
	printf("\n\n");
}
