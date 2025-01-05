#include "scene.h"

void Scene::SetCamera(Camera camera)
{
	this->camera = camera;
}


void Scene::SetLight(Light light)
{
	this->light = light;
}

Scene::Scene()
{
	light = Light();
	camera = Camera();
	texture = Texture(3, TEXTURE_WIDHT, TEXTURE_HEIGHT, 0);
}


void Texture::SetPixel(int x, int y, glm::vec3 color)
{
	int index = (y * width + x) * channels;
	data[index] = color.r;
	data[index + 1] = color.g;
	data[index + 2] = color.b;
}

void Scene::UpdateTextureGpu(unsigned char* dev_texture_data, float* dev_projection, float* dev_view, float* dev_camera_position, float* dev_light_position, int sphere_count,
	Node* dev_tree, float* dev_intersecion_points,float* dev_intersection_result, int* dev_parts)
{

	// Matrix that transforms from camera space to screen space
	mat4 view = glm::inverse(glm::lookAt(camera.position, camera.position + camera.direction, camera.up));
	mat4 projection = glm::inverse(glm::perspectiveFov(glm::radians(camera.fov), (float)texture.width, (float)texture.height, 0.1f, 10.0f));



	float projection2[16];
	float view2[16];

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			projection2[i * 4 + j] = projection[j][i];
			view2[i * 4 + j] = view[j][i];
		}
	}

	float camera_position[3] = {camera.position.x, camera.position.y, camera.position.z};
	float light_position[3] = { light.position.x, light.position.y, light.position.z };

	cudaMemcpy(dev_projection, projection2, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_view, view2, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_camera_position, camera_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_light_position, light_position, 3 * sizeof(float), cudaMemcpyHostToDevice);

	UpdateOnGPU(dev_texture_data, TEXTURE_WIDHT, TEXTURE_HEIGHT, sphere_count, dev_projection, dev_view, dev_camera_position, dev_light_position, dev_tree, dev_intersecion_points, dev_intersection_result, dev_parts);
}

