#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>

#include "cuda_runtime.h"

#include "Tree.h"

using namespace std;

struct ParseSphere
{
	string index;
	float x;
	float y;
	float z;
	float radius;
	float r;
	float g;
	float b;
};

struct ParseCube
{
	string index;
	float x;
	float y;
	float z;
	float width;
	float height;
	float depth;
	float r;
	float g;
	float b;
};

struct ParseNode
{
	string indexStr;
	string left;
	string right;
	string operation;
	int index;
};

class TreeParser
{
	string fileName;

	const int MAX_SHAPES = 256;

	int num_spheres = 0;
	int num_cubes = 0;
	int num_nodes = 0;

	vector<Node> nodes;
	vector<ParseNode> parse_nodes= vector<ParseNode>();
	vector<Cube> cubes = vector<Cube>(MAX_SHAPES);
	vector<Sphere> spheres = vector<Sphere>(MAX_SHAPES);

	bool CreateObjects();
public:
	TreeParser(string fileName);

	bool Parse();
};

