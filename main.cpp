#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include "defines.h"
#include "methods.h"
#include "wah_test.h"
#include <cstdio>
#include <climits>
#include <bitset>
#include <iostream>


int main() {
    Benchmark(&BallotSyncWAH, nullptr, 10, "Time tests");
    //BallotSyncWAH(nullptr);
    return 0;
}