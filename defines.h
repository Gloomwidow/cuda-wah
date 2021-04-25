// This file is safe to #define without any #ifndefs

#ifndef UINT
#define UINT unsigned int
#endif // !UINT
#ifndef ULONG
#define ULONG unsigned long long
#endif // !ULONG

#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif // !FULL_MASK

#ifndef UNIT_TESTING
#define UNIT_TESTING true
#endif // !FULL_MASK

#ifndef CUDA_CHECK
#define CUDA_CHECK(call, label)																		\
{																									\
	cudaError_t cudaStatus = call;																	\
	if (cudaStatus != cudaSuccess){																	\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);										\
		fprintf(stderr, "%s: %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));	\
		goto label;																					\
	}																								\
}
#endif // !CUDA_CHECK
