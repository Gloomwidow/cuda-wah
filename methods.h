#pragma once
#include "defines.h"
#include <string>
void WarmUp();
UINT* RemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* AtomicAddWAH(int data_size, UINT* input, int threads_per_block);

UINT* SharedMemWAH(int data_size, UINT* input, int threads_per_block);
void smem_iterate_unittests();
void smem_iterate_benchmark(int batch_reserve, int batch_pos, int batch_size, int threads_per_block, std::string data_filename, UINT* data);
void unrolled_smem_iterate_benchmark(int batch_reserve, int batch_pos, int batch_size, int threads_per_block, std::string data_filename, UINT* data);

//optimized method variants
UINT* OptimizedRemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* OptimizedAtomicAddWAH(int data_size, UINT* input, int threads_per_block);
