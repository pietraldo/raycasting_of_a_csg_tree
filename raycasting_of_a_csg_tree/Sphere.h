#pragma once
#include <glm/glm.hpp>

class Sphere
{
public:

	Sphere(glm::vec3 position, float radius, glm::vec3 color = glm::vec3(255,255,255)) : 
		position(position), radius(radius), color(color){}

	glm::vec3 position;
	float radius;
	glm::vec3 color;

	bool IntersectionPoint(glm::vec3 rayOrigin, glm::vec3 rayDirection, float& t1, float& t2);
	bool Contains(glm::vec3 point);
};

