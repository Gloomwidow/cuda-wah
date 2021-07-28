#pragma once
#include "defines.h"
#include <string>

UINT* CpuWAH(int data_size, UINT* data);
void Test(UINT* (*tested_function)(int, UINT*, int), int data_size, UINT* data, int expected_size, UINT* expected, std::string test_name);
void Benchmark(UINT* (*tested_function)(int, UINT*, int), int data_size, UINT* d_data, std::string bench_name, int threads_per_block);
void UnitTests(UINT* (*tested_function)(int, UINT*, int));

