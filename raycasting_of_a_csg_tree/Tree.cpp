#include "Tree.h"

Tree::Tree()
{
}

bool Tree::Contains(glm::vec3 point, Node* node)
{
	if (node->isLeaf)
	{
		return node->sphere->Contains(point);
	}
	else
	{
		bool left = Contains(point, node->left);
		bool right = Contains(point, node->right);
		return node->functionPtr(left, right);
	}
}

bool Tree::Union(bool a, bool b)
{
	return a || b;
}

bool Tree::Intersection(bool a, bool b)
{
	return a && b;
}

bool Tree::Subtraction(bool a, bool b)
{
	return a && !b;
}

void Tree::SetRoot(Node* root)
{
	this->root = root;
}