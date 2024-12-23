#pragma once


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "Constants.h"

using namespace glm;

struct Texture {
	GLuint id;
	int width;
	int height;
	int channels;
	std::vector<unsigned char> data;
};

struct Sphere
{
	vec3 position;
	float radius;
};

void SetPixel(Texture& texture, int x, int y, glm::vec3 color)
{
	int index = (y * texture.width + x) * texture.channels;
	texture.data[index] = color.r;
	texture.data[index + 1] = color.g;
	texture.data[index + 2] = color.b;
}

float remap(float a, float b, float t)
{
	return (t - a) / (b - a);
}
struct Camera
{
	vec3 position;
	vec3 direction;
	vec3 up = vec3(0, 1, 0);
	float yaw = -90;
	float pitch = 0;
};

Camera camera = { vec3(0, 0, 2), vec3(0, 0, 1) };
void UpdateTextureCpu(Texture& texture)
{
	float distance = SCR_WIDTH / 2;

	vector<Sphere> spheres;
	spheres.push_back({ vec3(0, 0, 7), 1.0f });
	spheres.push_back({ vec3(0, 0, 2), 0.5f });
	spheres.push_back({ vec3(4, 0, 2), 0.5f });
	spheres.push_back({ vec3(-4, 0, 2), 0.5f });

	float stepX = 2 / (float)TEXTURE_WIDHT;
	float stepY = 2 / (float)TEXTURE_HEIGHT;
	float aspectRatio = (float)TEXTURE_WIDHT / (float)TEXTURE_HEIGHT;

	vec3 forward = normalize(camera.direction);
	vec3 right = normalize(cross(forward, camera.up));
	vec3 up = normalize(cross(right, forward));

	float translationX = dot(camera.position, right);
	float translationY = dot(camera.position, up);
	float translationZ = dot(camera.position, forward);

	mat4 viewMatrix = mat4(
		vec4(right.x, up.x, -forward.x, 0),
		vec4(right.y, up.y, -forward.y, 0),
		vec4(right.z, up.z, -forward.z, 0),
		vec4(-translationX, -translationY, translationZ, 1)
	);

	for (int i = 0; i < TEXTURE_WIDHT; i+=2)
	{
		for (int j = 0; j < TEXTURE_HEIGHT; j+=2)
		{
			vec3 color = vec3(0, 0, 0);
			float closest = 1000000;
			for (int k = 0; k < spheres.size(); k++)
			{
				vec3 ray = vec3(i - TEXTURE_WIDHT / 2, j - TEXTURE_HEIGHT / 2, distance);
				ray = normalize(ray);

				vec4 spherePos = viewMatrix * vec4(spheres[k].position, 0);
				vec3 spherePos3 = vec3(spherePos.x, spherePos.y, spherePos.z);


				float t = dot(spherePos3 - camera.position, ray);
				vec3 p = camera.position + t * ray;
				float y = length(spherePos3 - p);
				if (y < spheres[k].radius)
				{
					float x = sqrt(spheres[k].radius * spheres[k].radius - y * y);
					float t1 = t - x;

					if (t1 < closest && t1>0)
					{
						closest = t1;
						float distanceToSphere = length(spherePos3 - camera.position);
						float c = remap(distanceToSphere + spheres[k].radius, distanceToSphere - spheres[k].radius, t1);
						if (k == 0)
							color = vec3(255 * c, 255 * c, 255 * c);
						else if (k == 1)
							color = vec3(c * 100, c * 100, 255 * c);
						else if (k == 2)
							color = vec3(255 * c, 200 * c, 200 * c);
						else if (k == 3)
							color = vec3(100 * c, 255 * c, 100 * c);
					}
				}
			}
			SetPixel(texture, i, j, color);
		}
	}
}
