#pragma once
#include <glm/glm.hpp>

class Light
{

public:
	bool rotateLight = false;

	Light() : position(glm::vec3(0, 0, 0)), color(glm::vec3(1, 1, 1)) {}
	Light(glm::vec3 position, glm::vec3 color) : position(position), color(color) {}
	glm::vec3 position;
	glm::vec3 color;

};