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
#include <vector>
#include <fstream>
#include <chrono>
#include "defines.h"
#include "bit_functions.cuh"


void CharTextBenchmark(long int batch_char_size, std::string data_filename);
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main() {
    if (UNIT_TESTING)
    {
        //UnitTests(&BallotSyncWAH);
        //UnitTests(&AtomicAddWAH);
	    UnitTests(&SharedMemWAH);
        // printf("próba mikrofonu\n");
		//UnitTests(&RemoveIfSharedMemWAH);
    }
    else
    {
        printf("Launching Warm Up Kernel...\n");
        WarmUp();
        printf("Starting tests data: ASCII texts\n");
        CharTextBenchmark(20000000, "database.txt");
    }
    return 0;
}

void RunWithBatch(int batch_reserve, int batch_pos, int batch_size, std::string data_filename, UINT* data)
{
    UINT* d_data;
    cudaMalloc((UINT**)&d_data, sizeof(UINT) * batch_reserve);
    cudaMemcpy(d_data, data, sizeof(UINT) * batch_reserve, cudaMemcpyHostToDevice);
    //Benchmark(&BallotSyncWAH, batch_reserve, d_data, 1, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve*32) + ";" + "remove_if;"+std::to_string(GPU_THREADS_COUNT)+";" + std::to_string(batch_size) + ";", false);
    //Benchmark(&AtomicAddWAH, batch_reserve, d_data, 1, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve*32) + ";" +  "atomicAdd;"+std::to_string(GPU_THREADS_COUNT)+";" + std::to_string(batch_size) + ";", false);
    Benchmark(&SharedMemWAH, batch_reserve, d_data, 1, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve * 32) + ";" + "sharedMem;" + std::to_string(GPU_THREADS_COUNT) + ";" + std::to_string(batch_size) + ";", false);
    cudaFree(d_data);
}

void CharTextBenchmark(long int batch_char_size, std::string data_filename)
{

    std::ofstream log("results_load.csv", std::ios_base::app | std::ios_base::out);
    std::fstream file("Data/" + data_filename, std::ios::binary | std::ios::in);
    if (!file)
    {
        printf("Cannot open file!\n");
        return;
    }

    long int batch_bit_size = batch_char_size * 8;
    long int batch_int_count = (long int)ceil((batch_bit_size*1.0f) / 31.0f);

    UINT* data = new UINT[batch_int_count];

    unsigned long long bit_pos = 0;
    int curr_char = 0;
    int batch_part_index = 0;
    auto start = Time::now();
    while (!file.eof())
    {
        // fill table with data from file
        while (curr_char < batch_char_size) 
        {
            char c;
            file.get(c);
            for (int i = 7; i >= 0; i--)
            {
                int uint_pos = bit_pos / 32;             
                if (bit_pos % 32 == 0) //first bit in UINT (WAH flag bit) should be 0 for algorithms 
                {
                    data[uint_pos] = clear_bit(data[uint_pos], 0);
                    bit_pos++;
                }
                int curr_bit = ((c >> i) & 1);
                int local_uint_pos = bit_pos % 32;
                if(curr_bit) data[uint_pos] = fill_bit(data[uint_pos], local_uint_pos);
                else data[uint_pos] = clear_bit(data[uint_pos], local_uint_pos);
                bit_pos++;
            }
            curr_char++;
        }
        //fill remaining places with 0 bits
        if (bit_pos < batch_bit_size)
        {
            int uint_pos = bit_pos / 32;
            int local_uint_pos = bit_pos % 32;
            data[uint_pos] = clear_bit(data[uint_pos], local_uint_pos);
            bit_pos++;
        } 
        auto end = Time::now();
        fsec fs = end - start; 
        log << data_filename + ";" + std::to_string(batch_part_index) + ";" + std::to_string(batch_bit_size) + ";" + std::to_string(fs.count()) + ";"<<std::endl;
        printf("Batch %d load done...\n", batch_part_index + 1);
        RunWithBatch(batch_int_count, batch_part_index, batch_bit_size, data_filename, data);
        printf("Batch %d CUDA done...\n", batch_part_index + 1);
        batch_part_index++;
        curr_char = 0;
        bit_pos = 0;
        start = Time::now();
    }
    //fill remaining places with 0 if no file data is left
    if (bit_pos < batch_bit_size)
    {
        int uint_pos = bit_pos / 32;
        int local_uint_pos = bit_pos % 32;
        data[uint_pos] = clear_bit(data[uint_pos], local_uint_pos);
        bit_pos++;
    }
    auto end = Time::now();
    fsec fs = end - start;
    log << data_filename + ";" + std::to_string(batch_part_index) + ";" + std::to_string(batch_bit_size) + ";" + std::to_string(fs.count()) + ";" << std::endl;
    printf("Batch %d load done...\n", batch_part_index + 1);
    RunWithBatch(batch_int_count, batch_part_index, batch_bit_size, data_filename, data);
    printf("Batch %d CUDA done...\n", batch_part_index + 1);
    batch_part_index++;
    curr_char = 0;
    bit_pos = 0;
    start = Time::now();

    file.close();
    log.close();
    delete[] data;
}
