#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

class Camera
{

public:
	vec3 position;
	vec3 direction;
	vec3 up = vec3(0, 1, 0);

	float yaw = 1;
	float pitch = 0;
	float fov = 45;

	float r = 20;

	bool animation = false;
	bool rotateScene = true;
	bool setCameraToCenter = false;
	
	int cameraDirection = 1; // 1 - up, -1 - down

	float speed = 10.0f;

	Camera()
	{
		position = vec3(0, 0, 0);
		direction = vec3(0, 0, 1);
	}
	Camera(vec3 position)
	{
		this->position = position;
		direction = vec3(0, 0, 1);
	}
	void UpdatePosition();
	~Camera() = default;
};

