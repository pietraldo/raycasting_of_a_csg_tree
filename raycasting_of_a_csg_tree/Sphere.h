#pragma once
#include <glm/glm.hpp>

class Sphere
{
public:

	Sphere(glm::vec3 position, float radius) : position(position), radius(radius) {}

	glm::vec3 position;
	float radius;

	bool IntersectionPoint(glm::vec3 rayOrigin, glm::vec3 rayDirection, float& t1, float& t2);
};