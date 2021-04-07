#include "DataGenerators.h"
#pragma once

class ZerosGenerator : public IntDataGenerator
{
public:
	ZerosGenerator()
	{

	}
	unsigned int* GetDeviceData(int blocks)
	{
		return nullptr;
	}
	
	unsigned int* GetHostData(int blocks)
	{
		unsigned int* result = new unsigned int[blocks];
		for (int i = 0; i < blocks; i++) result[i] = 0;
		return result;
	}
};
