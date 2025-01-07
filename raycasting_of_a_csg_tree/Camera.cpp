#include "Camera.h"

void Camera::UpdatePosition()
{
	if (rotateScene)
	{
		vec3 camPos = vec3(0, 0, -r);

		mat3 rotationY = mat3(
			vec3(cos(glm::radians(yaw)), 0, sin(glm::radians(yaw))),
			vec3(0, 1, 0),
			vec3(-sin(glm::radians(yaw)), 0, cos(glm::radians(yaw)))
		);

		mat3 rotationX = mat3(
			vec3(1, 0, 0),
			vec3(0, cos(glm::radians(pitch)), -sin(glm::radians(pitch))),
			vec3(0, sin(glm::radians(pitch)), cos(glm::radians(pitch)))
		);

		camPos = rotationY * rotationX * camPos;

		position = camPos;
		direction = -position;
		setCameraToCenter = true;
	}
	else
	{
		if (setCameraToCenter)
		{
			position = vec3(-11, -4, 0);
			pitch = 2;
			yaw = 22;
			setCameraToCenter = false;
		}

		direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		direction.y = sin(glm::radians(pitch));
		direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		direction = glm::normalize(direction);
	}
}