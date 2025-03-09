#include "Test.h"

void Test::test1()
{
	const int n = 12;
	float sphereIntervals[n] = { 80, 81, 1000,1000, -1,-1, 80.5, 87, -1,-1,-1,-1 };
	float temp[n];

	int parts[20] = { 8,9, 10,11, 4, 5,6,7,4,7,8,11,0,1,2,3,0,3,4,11 };
	for (int i = 0; i < 5; i++)
	{
		int p1 = parts[4 * i];
		int p2 = parts[4 * i + 2];
		int k1 = parts[4 * i + 1];
		int k2 = parts[4 * i + 3];

		std::cout << "before: ";
		for (int j = 0; j < n; j++)
		{
			std::cout << sphereIntervals[j] << " ";
		}
		std::cout << std::endl;

		AddIntervals2(sphereIntervals, temp, p1, p2, k1, k2, false);

		std::cout << "after: ";
		for (int j = 0; j < n; j++)
		{
			std::cout << sphereIntervals[j] << " ";
		}
		std::cout << std::endl << std::endl << std::endl;
	}


}

void Test::test2()
{
	const int n = 6;
	float sphereIntervals[n] = { 12,23,16.1, 16.3, 18,20 };
	float temp[n];


	int p1 = 0;
	int p2 = 2;
	int k1 = 1;
	int k2 = 5;

	std::cout << "before: ";
	for (int j = 0; j < n; j++)
	{
		std::cout << sphereIntervals[j] << " ";
	}
	std::cout << std::endl;

	CommonPartIntervals(sphereIntervals, temp, p1, p2, k1, k2, false);

	std::cout << "after: ";
	for (int j = 0; j < n; j++)
	{
		std::cout << sphereIntervals[j] << " ";
	}
	std::cout << std::endl << std::endl << std::endl;



}