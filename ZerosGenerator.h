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
		unsigned int* result = new unsigned int[blocks];
		for (int i = 0; i < blocks; i++) result[i] = 0;
		return result;
	}
	
	unsigned int* GetHostData(int blocks)
	{
		return nullptr;
	}
};
