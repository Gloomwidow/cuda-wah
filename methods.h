#pragma once
#ifndef WAH_FUN_TYPEDEF
#define WAH_FUN_TYPEDEF
typedef UINT* (*WAH_fun)(int size, UINT* input);
#endif // WAH_FUN_TYPEDEF

void WarmUp();
UINT* RemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* AtomicAddWAH(int data_size, UINT* input, int threads_per_block);
UINT* SharedMemWAH(int data_size, UINT* input, int threads_per_block);
UINT* RemoveIfSharedMemWAH(int size, UINT* input, int threads_per_block);

//optimized method variants
UINT* OptimizedRemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* OptimizedAtomicAddWAH(int data_size, UINT* input, int threads_per_block);

WAH_fun* get_wahs(int* count);