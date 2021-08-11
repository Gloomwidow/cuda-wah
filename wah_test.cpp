#include "wah_test.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bit_functions.cuh"
#include "defines.h"
#include <vector>
#include <fstream>
#include <string>
#define MISMATCH_MAX 0

int last_wah_count = 0;
int tests_threads_per_block = 1024;

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

void Test(UINT* (*tested_function)(int,UINT*,int),int data_size, UINT* data, int expected_size, UINT* expected, std::string test_name)
{
	int dsize = data_size;
	UINT* d_data;
	cudaMalloc((UINT**)&d_data, sizeof(UINT) * dsize);
	cudaMemcpy(d_data, data, sizeof(UINT) * dsize, cudaMemcpyHostToDevice);

	printf("TEST '%s' - Input Size: %u \n", test_name.c_str(), dsize);
	printf("=======================================================\n");
	UINT* actual = tested_function(dsize, d_data, tests_threads_per_block);
	// printf("expected outputSize: %d\n", expected_size);
	printf("=======================================================\n");
	cudaFree(d_data);
	if (actual == nullptr)
	{
		printf("\033[0;31mFAILED: function did not return any data!\033[0m\n");
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
				printf("Tables' values mismatch on position %d!\n", i);
				printf("\tExpected: "); printBits(sizeof(UINT), &expected[i]);
				printf("\tActual:   "); printBits(sizeof(UINT), &actual[i]);
			}
		}
	}
	delete[] actual;
	if (mismatches == 0) printf("\033[0;32mPASSED\033[0m\n");
	else printf("\033[0;31mFAILED: Found mismatches. Count: %d\033[0m\n", mismatches);
	
}


void Benchmark(UINT* (*tested_function)(int, UINT*, int), int data_size, UINT* d_data, std::string bench_name, int threads_per_block)
{
	int size = data_size;

	std::ofstream log("results_"+std::to_string(threads_per_block)+".csv", std::ios_base::app | std::ios_base::out);

	cudaEvent_t start, stop;
	float max = -1;
	float mean = 0;
	float min = 0; 
	float exec_time = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	UINT * result = tested_function(size,d_data,threads_per_block);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&exec_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	exec_time /= 1000;
	std::string resultRow;
	resultRow += bench_name;
	std::string timeString = std::to_string(exec_time);
	std::replace(timeString.begin(), timeString.end(), '.', ',');
	resultRow += timeString;
	resultRow += ";";
	log << resultRow << std::endl;
	delete[] result;

	log.close();
}



void NoCompressTest(UINT* (*tested_function)(int, UINT*, int))
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

void AllCompressTest(UINT* (*tested_function)(int, UINT*, int))
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

void BetweenWarpsMerge(UINT* (*tested_function)(int, UINT*, int))
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

void MixedCompressions(UINT* (*tested_function)(int, UINT*, int))
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

void MixedCompressionWithLiterals(UINT* (*tested_function)(int, UINT*, int))
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

void AllVariants_1(UINT* (*tested_function)(int, UINT*, int))
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

void MultipleWarpsMerge(UINT* (*tested_function)(int, UINT*, int))
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

void BigMerge(UINT* (*tested_function)(int, UINT*, int))
{
	int offset = 5;
	int length = 12;
	int warps = 256;
	int size = 32 * warps;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 0;
	}
	for (int i = 0; i < warps-2; i++)
	{
		int start = 32 * (i+1) - offset;
		for (int j = start; j < start+length; j++)
		{
			table[j] = 1;
		}
	}
	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Merge");
	delete[] table;
	delete[] result;
}

void BigMergeWithInterrupts(UINT* (*tested_function)(int, UINT*, int))
{
	int warps = 512;
	int size = 32 * warps;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 0;
	}
	table[11] = 11;
	table[1353] = 1353;
	table[5600] = 5600;
	table[10000] = 10000;
	table[12765] = 12765;
	
	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Merge With Interrupts");
	delete[] table;
	delete[] result;
}

void BigNoMerging(UINT* (*tested_function)(int, UINT*, int))
{
	int warps = 512;
	int size = 32 * warps;
	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 0;
	}
	for (int i = 0; i < warps - 2; i++)
	{
		int start = 31 * (i + 1);
		table[start] = start;
	}

	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Without Merging");
	delete[] table;
	delete[] result;
}

void BigMultipleDifferentSequences(UINT* (*tested_function)(int, UINT*, int))
{
	int warps = 512;
	int size = 32 * warps;
	int seqs = 4;
	int lengths[4]{ 68,111,36,245 };
	int sel_l = 0;
	int symbol = 0;
	int curr_length = 0;

	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = symbol;
		curr_length++;
		if (curr_length == lengths[sel_l])
		{
			sel_l = (sel_l + 1) % seqs;
			curr_length = 0;
			if (symbol == 0) symbol = 0x7FFFFFFF;
			else symbol = 0;
		}
	}

	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Multiple Different Sequences");
	delete[] table;
	delete[] result;
}

void BigMultipleDifferentSequencesWithLiterals(UINT* (*tested_function)(int, UINT*, int))
{
	int warps = 1024;
	int size = 32 * warps;
	int seqs = 9;
	int lengths[9]{ 68,321,4,323,43,22,9,213,55 };
	int sel_l = 0;
	int symbol = 0;
	int curr_length = 0;

	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = symbol;
		curr_length++;
		if (curr_length == lengths[sel_l])
		{
			sel_l = (sel_l + 1) % seqs;
			curr_length = 0;
			if (symbol == 0) symbol = 0x7FFFFFFF;
			else if (symbol == 0x7FFFFFFF) symbol = 256;
			else symbol = 0;
		}
	}

	UINT* result = CpuWAH(size, table);

	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Multiple Different Sequences With Literals");
	delete[] table;
	delete[] result;
}

void BigScarceSequences(UINT* (*tested_function)(int, UINT*, int))
{
	int warps = 1024;
	int size = 32 * warps;
	int seqs = 9;
	int sel_l = 0;
	int symbol = 0;
	int curr_length = 0;

	UINT* table = new UINT[size];
	for (int i = 0; i < size; i++)
	{
		table[i] = 556;	
	}

	for (int i = 129; i < 134; i++)
	{
		table[i] = 0;
	}
	for (int i = 789; i < 798; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	for (int i = 2556; i < 2559; i++)
	{
		table[i] = 0x7FFFFFFF;
	}
	for (int i = 5899; i < 5912; i++)
	{
		table[i] = 0;
	}
	for (int i = 12000; i < 12009; i++)
	{
		table[i] = 0;
	}
	for (int i = 26055; i < 26069; i++)
	{
		table[i] = 0x7FFFFFFF;
	}

	UINT* result = CpuWAH(size, table);
	int test_result_size = last_wah_count;
	Test(tested_function, size, table, test_result_size, result, "Big Scarce Sequences");
	delete[] table;
	delete[] result;
}

void UnitTests(UINT* (*tested_function)(int, UINT*, int))
{
	printf("Performing unit tests...\n");
	NoCompressTest(tested_function);
	AllCompressTest(tested_function);
	BetweenWarpsMerge(tested_function);
	MixedCompressions(tested_function);
	MixedCompressionWithLiterals(tested_function);
	AllVariants_1(tested_function);
	MultipleWarpsMerge(tested_function);

	BigMerge(tested_function);
	BigMergeWithInterrupts(tested_function);
	BigNoMerging(tested_function);
	BigMultipleDifferentSequences(tested_function);
	BigMultipleDifferentSequencesWithLiterals(tested_function);
	BigScarceSequences(tested_function);
	printf("\n\n");
}
