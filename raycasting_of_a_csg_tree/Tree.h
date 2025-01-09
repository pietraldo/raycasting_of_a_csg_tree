#pragma once

struct Cube
{
	float3 vertices[8];
};

struct Sphere
{
	float radius;
	float3 position;
	int3 color;
};

struct Node
{
	int left;
	int right;
	int parent;
	Sphere* sphere;
	int operation;
};

