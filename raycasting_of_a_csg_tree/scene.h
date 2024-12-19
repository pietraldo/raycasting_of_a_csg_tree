#pragma once


#include <glm/glm.hpp>
#include <vector>

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
vec3 camera = vec3(0, 0, 0);
float angle = 0;
void UpdateTextureCpu(Texture& texture)
{
	int distance = 400;

	mat3 rotation = mat3(cos(angle), 0, sin(angle), 0, 1, 0, -sin(angle), 0, cos(angle));

	

	vector<Sphere> spheres;
	spheres.push_back({ vec3(0, 0, 7), 1.0f });
	spheres.push_back({ vec3(0, 0, 2), 0.5f });

	for (int i = 0; i < 800; i++)
	{
		for (int j = 0; j < 600; j++)
		{
			vec3 color = vec3(0, 0, 0);
			float closest = 1000000;
			for (int k = 0; k < spheres.size(); k++)
			{
				vec3 ray = vec3(i - 400, j - 300, distance);
				ray = normalize(ray);
				ray = rotation * ray;

				float t = dot(spheres[k].position - camera, ray);
				vec3 p = camera + t * ray;
				float y = length(spheres[k].position - p);
				if (y < spheres[k].radius)
				{
					float x = sqrt(spheres[k].radius*spheres[k].radius - y * y);
					float t1 = t - x;

					if (t1 < closest)
					{
						closest = t1;
						float distanceToSphere = length(spheres[k].position - camera);
						float c = remap(distanceToSphere + spheres[k].radius, distanceToSphere - spheres[k].radius, t1);
						color = vec3(c*100, c*100, 255*c);
					}

				}

			}
			SetPixel(texture, i, j, color);
		}
	}
}