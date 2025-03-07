#include "Test.h"

void Test::test1()
{
	const int n = 4;
	float sphereIntervals[n] = { 1, 2, 2, 4 };
	float temp[n];
	int p1 = 0;
	int k1 = 1;
	int p2 = 2;
	int k2 = 3;

	std::cout << "before: ";
	for (int i = 0; i < n; i++)
	{
		std::cout << sphereIntervals[i] << " ";
	}
	std::cout << std::endl;

	AddIntervals2(sphereIntervals, temp, p1, p2, k1, k2, false);

	std::cout<<"after: ";
	for (int i = 0; i < n; i++)
	{
		std::cout << sphereIntervals[i]<<" ";
	}
	std::cout << std::endl;
}