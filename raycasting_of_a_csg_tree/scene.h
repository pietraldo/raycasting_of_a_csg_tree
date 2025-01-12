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
#include "Light.h"
#include "Texture.h"
#include "kernels.cuh"
#include "Tree.h"
#include "GPUdata.h"

using namespace glm;
using namespace std;

class Scene
{
	
	Light light;
	Camera camera;
	Texture texture;

public:

	float angle = 0;
	float camera_rotation = 0.25;
	float light_rotation = 0.03;

	Scene();
	void SetCamera(Camera camera);
	void SetLight(Light light);
	void SetLightPosition(vec3 light_pos);

	Camera& GetCamera() { return camera; }
	Light& GetLight() { return light; }
	Texture& GetTexture() { return texture; }

	void Update();

	void UpdateTextureGpu(GPUdata& data);
};










