#pragma once
#include "defines.h"
void WarmUp();
UINT* RemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* AtomicAddWAH(int data_size, UINT* input, int threads_per_block);
UINT* SharedMemWAH(int data_size, UINT* input, int threads_per_block);

//optimized method variants
UINT* OptimizedRemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* OptimizedAtomicAddWAH(int data_size, UINT* input, int threads_per_block);
