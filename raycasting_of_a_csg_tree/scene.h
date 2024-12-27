#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <vector>

#include "Constants.h"
#include "Camera.h"
#include "Sphere.h"
#include "Light.h"
#include "Texture.h"


using namespace glm;
using namespace std;

class Scene
{
	vector<Sphere> spheres;
	Light light;
	Camera camera;
	Texture texture;

public:
	Scene();
	void SetCamera(Camera camera);
	void AddSphere(Sphere sphere);
	void SetLight(Light light);

	Camera& GetCamera() { return camera; }
	Texture& GetTexture() { return texture; }
	void UpdateTextureCpu();
};










