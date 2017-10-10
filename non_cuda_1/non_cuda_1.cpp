// non_cuda_1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

int main(void)
{
	bool *test = new bool[40];

	for (int i = 0; i < 10; i++) {
		test[i] = (bool)(i % 3);
		std::cout << "i: " << i << " : " << test[i] << std::endl;
	}

	int *foo;
	for (int i = 0; i < 10; i++)
	{
		foo = (int *)(test[i]);
	}

	for (int i = 0; i < 10; i++)
	{
		test[i] = (int)(((int)test[i]) ^ 0x01);
	}
	return 0;
}


