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

using namespace glm;
using namespace std;

class Scene
{
	
	Light light;
	Camera camera;
	Texture texture;

public:

	float angle = 0;

	Scene();
	void SetCamera(Camera camera);
	void SetLight(Light light);
	void SetLightPosition(vec3 light_pos);

	Camera& GetCamera() { return camera; }
	Light& GetLight() { return light; }
	Texture& GetTexture() { return texture; }

	void UpdateTextureGpu(unsigned char* dev_tecture_data, float* dev_projection, float* dev_view, 
		float* dev_camera_position, float* dev_light_position, int sphere_count, 
		Node* dev_tree, float* dev_intersection_result, 
		int* dev_parts,  Sphere* dev_spheres, Cube* dev_cubes);
};










