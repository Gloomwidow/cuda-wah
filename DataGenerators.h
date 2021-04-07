#pragma once
#ifndef UINT
#define UINT unsigned int
#endif // !UINT

#ifndef ULONG
#define ULONG unsigned long long
#endif // !ULONG

class IntDataGenerator 
{
public:
	IntDataGenerator()
	{

	}
	virtual UINT* GetDeviceData(int blocks) = 0;
	virtual UINT* GetHostData(int blocks) = 0;
};

class LongIntDataGenerator
{
public:
	LongIntDataGenerator()
	{

	}
	virtual ULONG* GetDeviceData(int blocks) = 0;
	virtual ULONG* GetHostData(int blocks) = 0;
};

