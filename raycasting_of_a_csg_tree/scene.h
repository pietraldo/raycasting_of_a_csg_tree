#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <vector>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "Constants.h"
#include "Camera.h"
#include "Sphere.h"
#include "Light.h"
#include "Texture.h"
#include "kernels.cuh"

using namespace glm;
using namespace std;

class Scene
{
	
	Light light;
	Camera camera;
	Texture texture;

public:
	vector<Sphere> spheres;

	Scene();
	void SetCamera(Camera camera);
	void AddSphere(Sphere sphere);
	void SetLight(Light light);

	Camera& GetCamera() { return camera; }
	Texture& GetTexture() { return texture; }
	void UpdateTextureCpu();
	void UpdateTextureGpu(unsigned char* dev_tecture_data, DevSphere* dev_spheres, float* dev_projection, float* dev_view, float* dev_camera_position, float* dev_light_position, int sphere_count);
};










