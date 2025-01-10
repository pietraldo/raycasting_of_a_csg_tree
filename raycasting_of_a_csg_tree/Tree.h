#pragma once

struct Cube
{
	float3 vertices[8];
	int3 color;
};

struct Sphere
{
	float radius;
	float3 position;
	int3 color;
};

struct Cylinder
{
	float radius;
	float height;
	float3 position;
	float3 axis;
	int3 color;
};

struct Node
{
	int left;
	int right;
	int parent;
	int shape;
	Sphere* sphere; // shape = 1
	Cube* cube; // shape = 2
	Cylinder* cylinder; // shape = 3
	char operation;
};

