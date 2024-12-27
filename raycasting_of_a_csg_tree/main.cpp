
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels.cuh"
#include "scene.h"
#include "Camera.h"
#include "Constants.h"
#include "Window.h"

// Function declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
int createWindow(GLFWwindow*& window);
void InitImGui(GLFWwindow* window);
void processInput(GLFWwindow* window, float dt);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void renderWindow(GLFWwindow* window, Scene& scene);


float lastX = TEXTURE_WIDHT / 2, lastY = TEXTURE_HEIGHT / 2;


void RegisterTexture(Texture& texture) {
	// Generate and bind the texture
	glGenTextures(1, &texture.id);
	glBindTexture(GL_TEXTURE_2D, texture.id);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Determine the appropriate OpenGL format
	GLenum format = (texture.channels == 3) ? GL_RGB : GL_RGBA;

	// Upload texture data to the GPU
	glTexImage2D(GL_TEXTURE_2D, 0, format, texture.width, texture.height, 0, format, GL_UNSIGNED_BYTE, texture.data.data());

	// Unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);
}

Scene scene;
// Main function
int main() {

	Window window(SCR_WIDTH, SCR_HEIGHT);

	//glfwSetScrollCallback(window, scroll_callback);


	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, 0)));
	scene.AddSphere(Sphere(vec3(0, 0, 7), 1.0f));
	scene.AddSphere(Sphere(vec3(0, 0, 2), 0.5f));
	scene.AddSphere(Sphere(vec3(0, 0, 4), 0.5f));
	scene.AddSphere(Sphere(vec3(0, 0, 6), 0.5f));
	scene.AddSphere(Sphere(vec3(0, 0, 8), 0.5f));
	scene.AddSphere(Sphere(vec3(4, 3, -4), 0.5f));
	scene.AddSphere(Sphere(vec3(-4, 3, -4), 0.5f));
	scene.AddSphere(Sphere(vec3(4, 3, 4), 0.5f));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	RegisterTexture(scene.GetTexture());

	

	auto last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {
		
		auto time = glfwGetTime();
		float dt = time - last;
		std::cout << 1 / dt << std::endl;

		last = time;
		
		window.ProccessInput(scene, dt);

		float r = 10.0f;
		scene.SetLight(Light(vec3(r * cos(time), 0, r * sin(time)), vec3(1,1,1)));

		scene.UpdateTextureCpu();

		//clear the screen
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.Render(scene);
		

		// Swap buffers and poll event
		glfwSwapBuffers(window.GetWindow());
		glfwPollEvents();
	}

	return 0;
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	Camera& camera = scene.GetCamera();
	camera.fov -= (float)yoffset;
	if(camera.fov>90)
		camera.fov = 90;
	if (camera.fov < 1)
		camera.fov = 1;
}