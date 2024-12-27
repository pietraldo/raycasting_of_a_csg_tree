#pragma once

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <vector>

class Texture{
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

	void SetPixel(int x, int y, glm::vec3 color);
};
