#pragma once
#include "defines.h"
#include <string>

void Test(UINT* (*tested_function)(UINT*), UINT* data, UINT* expected, std::string test_name);
void Benchmark(UINT* (*tested_function)(UINT*), UINT* data, int repeats, std::string bench_name, bool print_times = false);

