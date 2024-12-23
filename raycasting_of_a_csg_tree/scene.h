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
	float yaw = 90;
	float pitch = 0;
	float fov = 45;
};

Camera camera = { vec3(0, 0, 0), vec3(0, 0, 1) };
void UpdateTextureCpu(Texture& texture)
{
	float distance = SCR_WIDTH / 2;

	vector<Sphere> spheres;
	spheres.push_back({ vec3(0, 0, 7), 1.0f });
	spheres.push_back({ vec3(0, 0, 2), 0.5f });
	spheres.push_back({ vec3(4, 0, 2), 0.5f });
	spheres.push_back({ vec3(-4, 0, 2), 0.5f });


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

	// Matrix that transforms from camera space to screen space
	mat4 view = glm::inverse(glm::lookAt(camera.position, camera.position + camera.direction, camera.up));
	mat4 projection = glm::inverse(glm::perspectiveFov(glm::radians(camera.fov), (float)TEXTURE_WIDHT, (float)TEXTURE_HEIGHT, 0.1f, 10.0f));

	float stepX = 2 / (float)TEXTURE_WIDHT;
	float stepY = 2 / (float)TEXTURE_HEIGHT;
	for (int i = 0; i < TEXTURE_WIDHT; i+=2)
	{
		for (int j = 0; j < TEXTURE_HEIGHT; j+=2)
		{
			vec3 ray = vec3(-1 + i * stepX, -1 + j * stepY, 1.0f);
			vec4 target = projection * vec4(ray, 1.0f);
			target = normalize(target / target.w);
			target.w = 0.0f;

			ray = view * target;

			vec3 color = vec3(0, 0, 0);
			float closest = 1000000;
			for (int k = 0; k < spheres.size(); k++)
			{
				

				float t = dot(spheres[k].position - camera.position, ray);
				vec3 p = camera.position + t * ray;
				float y = length(spheres[k].position - p);
				if (y < spheres[k].radius)
				{
					float x = sqrt(spheres[k].radius * spheres[k].radius - y * y);
					float t1 = t - x;

					if (t1 < closest && t1>0)
					{
						closest = t1;
						float distanceToSphere = length(spheres[k].position - camera.position);
						float c = remap(distanceToSphere + spheres[k].radius, distanceToSphere - spheres[k].radius, t1);
						if(c>1)
							c = 1;
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
