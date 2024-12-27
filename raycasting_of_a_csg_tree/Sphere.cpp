#include "Sphere.h"

bool Sphere::IntersectionPoint(glm::vec3 rayOrigin, glm::vec3 rayDirection, float& t1, float& t2)
{
	float a = dot(rayDirection, rayDirection);
	float b = 2 * dot(rayDirection, rayOrigin - position);
	float c = dot(rayOrigin - position, rayOrigin - position) - radius * radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
	{
		return false;
	}
	t1 = (-b - sqrt(discriminant)) / (2 * a);
	t2 = (-b + sqrt(discriminant)) / (2 * a);
	return true;
}

const float EPSILON = 0.0001;
bool Sphere::Contains(glm::vec3 point)
{
	return length(point - position) < radius;
}