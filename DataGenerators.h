#pragma once
#define UInt unsigned int
#define ULong unsigned long int

class IntDataGenerator 
{
public:
	IntDataGenerator()
	{

	}
	virtual UInt* GetDeviceData(int blocks) = 0;
	virtual UInt* GetHostData(int blocks) = 0;
};

class LongIntDataGenerator
{
public:
	LongIntDataGenerator()
	{

	}
	virtual ULong* GetDeviceData(int blocks) = 0;
	virtual ULong* GetHostData(int blocks) = 0;
};

