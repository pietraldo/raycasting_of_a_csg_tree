#pragma once

#include "Tree.h"

struct GPUdata
{
	unsigned char* dev_texture_data;
	float* dev_projection;
	float* dev_view;
	float* dev_camera_position;
	float* dev_light_postion;
	Node* dev_tree;
	int* dev_parts;
	Sphere* dev_spheres;
	Cube* dev_cubes;
	Cylinder* dev_cylinders;
	float* dev_intersection_result;
	int ShapeCount;
};