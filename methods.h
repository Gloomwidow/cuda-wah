#pragma once
void WarmUp();
UINT* BallotSyncWAH(int data_size, UINT* input);
UINT* AtomicAddWAH(int data_size, UINT* d_input);
UINT* SharedMemWAH(int size, UINT* input);