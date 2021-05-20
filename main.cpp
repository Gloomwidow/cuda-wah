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

extern void SharedMemWAH(UINT* input);// , size_t size);
void CharTextBenchmark();
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> fsec;


int main() {
	//SharedMemWAH(nullptr);//, 2);
   
    if (UNIT_TESTING)
    {
        UnitTests(&BallotSyncWAH);
        UnitTests(&AtomicAddWAH);
    }
    else
    {
        printf("Launching Warm Up Kernel...\n");
        WarmUp();
        printf("Starting tests data: ASCII texts\n");
        CharTextBenchmark();
    }
    return 0;
}

void RunWithBatch(int batch_reserve, int batch_pos, int batch_size, std::string data_filename, UINT* data)
{
    UINT* d_data;
    cudaMalloc((UINT**)&d_data, sizeof(UINT) * batch_reserve);
    cudaMemcpy(d_data, data, sizeof(UINT) * batch_reserve, cudaMemcpyHostToDevice);
    Benchmark(&BallotSyncWAH, batch_reserve, d_data, 1, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve*32) + ";" + "remove_if;"+std::to_string(GPU_THREADS_COUNT)+";" + std::to_string(batch_size) + ";", false);
    Benchmark(&AtomicAddWAH, batch_reserve, d_data, 1, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve*32) + ";" +  "atomicAdd;"+std::to_string(GPU_THREADS_COUNT)+";" + std::to_string(batch_size) + ";", false);
    cudaFree(d_data);
}

void CharTextBenchmark()
{

    std::string data_filename = "database.txt";
    std::ofstream log("results_load.csv", std::ios_base::app | std::ios_base::out);
    std::fstream file("Data/" + data_filename, std::ios::binary | std::ios::in);
    if (!file)
    {
        printf("Cannot open file!\n");
        return;
    }

    
    
    //printf("%lld\n", file_size);

    long int batch_char_size = 50000000;
    


    int batch_pos = 0;
    int curr_batch_size = 0;
    unsigned long long bit_pos = 0;

    std::vector<int> bits;
    int local_bit = 0;
    
    auto start = Time::now();
    while (!file.eof())
    {
        if (curr_batch_size < batch_char_size)
        {
            char c;
            file.get(c);
            for (int i = 7; i >= 0; i--)
            {
                if (local_bit == 0)
                {
                    bits.push_back(0);
                    local_bit++;
                }
                int bit = ((c >> i) & 1);
                //printf("%d", bit);
                bits.push_back(bit);
                local_bit = (local_bit + 1) % 32;
            }
            //printf("\n");
            curr_batch_size++;
        }
        else
        {
            int rem = bits.size() % 32;
            int to_fill = rem!=0?32-rem:0;
            //printf("%d, %d, to fill: %d\n",bits.size(), bits.size()/32, to_fill);
            for (int i = 0; i < to_fill; i++)
            {
                bits.push_back(0);
            }
            int data_size = bits.size() / 32;
            UINT* data = new UINT[data_size];
            
            for (int i = 0; i < data_size; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    if (bits[i*32+j]) data[i] = fill_bit(data[i], j);
                    else data[i] = clear_bit(data[i], j);
                }
            }
            auto end = Time::now();
            fsec fs = end - start; 
            log << data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_char_size * 8) + ";" + std::to_string(fs.count()) + ";"<<std::endl;
            printf("Batch %d load done...\n", batch_pos + 1);
            RunWithBatch(bits.size() / 32, batch_pos, batch_char_size * 8, data_filename, data);
            printf("Batch %d CUDA done...\n", batch_pos + 1);
            bits.clear();         
            batch_pos++;
            curr_batch_size = 0;
            delete[] data;
            start = Time::now();
        }
    }   
    if (curr_batch_size < batch_char_size && curr_batch_size>0)
    {
        while (curr_batch_size < batch_char_size)
        {
            char c = 0;
            for (int i = 7; i >= 0; i--)
            {
                if (local_bit == 0)
                {
                    bits.push_back(0);
                    local_bit++;
                }
                int bit = ((c >> i) & 1);
                bits.push_back(bit);
                local_bit = (local_bit + 1) % 32;
            }
            //printf("\n");
            curr_batch_size++;
        }
        int rem = bits.size() % 32;
        int to_fill = rem != 0 ? 32 - rem : 0;
        //printf("%d, %d, to fill: %d\n",bits.size(), bits.size()/32, to_fill);
        for (int i = 0; i < to_fill; i++)
        {
            bits.push_back(0);
        }
        int data_size = bits.size() / 32;
        UINT* data = new UINT[data_size];

        for (int i = 0; i < data_size; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                if (bits[i * 32 + j]) data[i] = fill_bit(data[i], j);
                else data[i] = clear_bit(data[i], j);
            }
        }
        auto end = Time::now();
        fsec fs = end - start;
        log << data_filename + ";"+std::to_string(batch_pos)+";"+ std::to_string(batch_char_size * 8) + ";" + std::to_string(fs.count()) + ";" << std::endl;
        RunWithBatch(bits.size() / 32, batch_pos, batch_char_size * 8, data_filename, data);
        bits.clear();
        batch_pos++;
        curr_batch_size = 0;
        delete[] data;
    }
    file.close();
    log.close();
}
