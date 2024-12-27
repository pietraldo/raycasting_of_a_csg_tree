
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



void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
float GetTimePassed(float& last);



Scene scene;
// Main function
int main() {

	Window window(SCR_WIDTH, SCR_HEIGHT);

	glfwSetScrollCallback(window.GetWindow(), scroll_callback);

	
	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, 0)));
	scene.AddSphere(Sphere(vec3(0, 0, 7), 1.0f));
	scene.AddSphere(Sphere(vec3(0, 0, 2), 0.5f, vec3(255, 0, 0)));
	scene.AddSphere(Sphere(vec3(0, 0, 4), 0.5f));
	scene.AddSphere(Sphere(vec3(0, 0, 6), 0.5f));
	scene.AddSphere(Sphere(vec3(0, 0, 8), 0.5f, vec3(0, 255, 0)));
	scene.AddSphere(Sphere(vec3(4, 3, -4), 0.5f));
	scene.AddSphere(Sphere(vec3(-4, 3, -4), 0.5f, vec3(255, 0, 255)));
	scene.AddSphere(Sphere(vec3(4, 3, 4), 0.5f, vec3(0, 0, 255)));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	window.RegisterTexture(scene.GetTexture());

	

	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {
		
		float dt = GetTimePassed(last);
		
		window.ProccessInput(scene, dt);

		float r = 10.0f;
		scene.SetLight(Light(vec3(r * cos(glfwGetTime()), 0, r * sin(glfwGetTime())), vec3(1,1,1)));

		scene.UpdateTextureCpu();

		window.ClearScreen();

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.Render(scene);
	}

	return 0;
}
float GetTimePassed(float& last) {
	auto time = glfwGetTime();
	float dt = time - last;
	last = time;
	std::cout << 1 / dt << std::endl;
	return dt;
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