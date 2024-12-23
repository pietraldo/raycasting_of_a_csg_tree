#include "scene.h"

void Scene::SetCamera(Camera camera)
{
	this->camera = camera;
}

void Scene::AddSphere(Sphere sphere)
{
	spheres.push_back(sphere);
}

void Scene::AddLight(Light light)
{
	lights.push_back(light);
}

Scene::Scene()
{
	camera = Camera(vec3(0, 0, 0));
	texture = Texture(3, TEXTURE_WIDHT, TEXTURE_HEIGHT, 0);
}


void Texture::SetPixel(int x, int y, glm::vec3 color)
{
	int index = (y * width + x) * channels;
	data[index] = color.r;
	data[index + 1] = color.g;
	data[index + 2] = color.b;
}

void Scene::UpdateTextureCpu()
{
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
	mat4 projection = glm::inverse(glm::perspectiveFov(glm::radians(camera.fov), (float)texture.width, (float)texture.height, 0.1f, 10.0f));

	float stepX = 2 / (float)texture.width;
	float stepY = 2 / (float)texture.height;
	for (int i = 0; i < texture.width; i += 2)
	{
		for (int j = 0; j < texture.height; j += 2)
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
						if (c > 1)
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
			texture.SetPixel(i, j, color);
		}
	}
}