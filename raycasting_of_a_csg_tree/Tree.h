#pragma once

#include "glm/glm.hpp"
#include "Sphere.h"

struct Node
{
	bool isLeaf;
	Node* left;
	Node* right;
	Sphere* sphere;
	bool (*functionPtr)(bool a, bool b);
};

class Tree
{
private:
	Node* root;
public:
	Tree();
	bool Contains(glm::vec3 point, Node* node);
	void SetRoot(Node* root);
	Node* GetRoot() { return root; }

	static bool Union(bool a, bool b);
	static bool Intersection(bool a, bool b);
	static bool Subtraction(bool a, bool b);
};

