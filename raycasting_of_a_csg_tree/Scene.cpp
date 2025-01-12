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

void Scene::Update()
{
	float r = 100000.0f;

	if (light.rotateLight)
	{
		angle += light_rotation;
	}
	
	SetLightPosition(vec3(r * cos(angle), 0, r * sin(angle)));

	if (camera.animation)
	{
		camera.yaw += camera_rotation;
		camera.pitch += camera.cameraDirection * 0.05;
		if (camera.pitch > 20)
		{
			camera.cameraDirection = -1;
		}
		if (camera.pitch < -20)
		{
			camera.cameraDirection = 1;
		}
	}
}

void Texture::SetPixel(int x, int y, glm::vec3 color)
{
	int index = (y * width + x) * channels;
	data[index] = color.r;
	data[index + 1] = color.g;
	data[index + 2] = color.b;
}

void Scene::UpdateTextureGpu(GPUdata& data)
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

	cudaMemcpy(data.dev_projection, projection2, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_view, view2, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_camera_position, camera_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_light_postion, light_position, 3 * sizeof(float), cudaMemcpyHostToDevice);

	UpdateOnGPU(data, TEXTURE_WIDHT, TEXTURE_HEIGHT);
}

void Scene::SetLightPosition(vec3 light_pos)
{
	light.position = light_pos;
}