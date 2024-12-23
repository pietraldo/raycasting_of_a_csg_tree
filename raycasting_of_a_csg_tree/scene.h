#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "Constants.h"
#include "Camera.h"

using namespace glm;
using namespace std;

class Light
{
public:
	Light() : position(vec3(0, 0, 0)), color(vec3(1, 1, 1)) {}
	Light(vec3 position, vec3 color) : position(position), color(color) {}
	vec3 position;
	vec3 color;

};

class Sphere
{
public:

	Sphere(vec3 position, float radius) : position(position), radius(radius) {}

	vec3 position;
	float radius;
};

class Texture {
public:
	GLuint id;
	int width;
	int height;
	int channels;
	std::vector<unsigned char> data;

	Texture() : id(0), width(0), height(0), channels(0), data(0) {};

	Texture(int channels, int width, int heigh, const unsigned char color) :
		id(0),
		channels(channels),
		width(width),
		height(heigh),
		data(std::vector<unsigned char>(width* heigh* channels, color)) {};

	void SetPixel(int x, int y, vec3 color);
};


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










