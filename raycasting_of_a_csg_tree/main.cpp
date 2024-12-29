
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
#include "DevStruct.h"
#include "Tree.h"


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
float GetTimePassed(float& last);


#include "kernels.cuh"

Scene scene;
// Main function
int main() {




	Window window(SCR_WIDTH, SCR_HEIGHT);

	glfwSetScrollCallback(window.GetWindow(), scroll_callback);

	const int SPHERE_COUNT = 6;
	const int NODE_COUNT = 2 * SPHERE_COUNT - 1;

	Sphere spheres_scene[SPHERE_COUNT];

	//const int sphere_render_radius = 10;
	/*for (int i = 0; i < SPHERE_COUNT; i++) {
		spheres_scene[i] = Sphere(vec3(rand() % sphere_render_radius - sphere_render_radius/2, rand() % sphere_render_radius - sphere_render_radius/2, rand() % sphere_render_radius - sphere_render_radius/2), 1.0f);
	}*/
	spheres_scene[0] = Sphere(vec3(1, 0, 0), 1.0f);
	spheres_scene[1] = Sphere(vec3(-1, 0, 0), 1.0f);
	spheres_scene[2] = Sphere(vec3(0, 1, 0), 1.0f);
	spheres_scene[3] = Sphere(vec3(0, -1, 0), 1.0f);
	spheres_scene[4] = Sphere(vec3(0, 0, 0), 1.5f);
	spheres_scene[5] = Sphere(vec3(0.5, 0, -1), 0.5f);

	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, -8)));
	for (int i = 0; i < SPHERE_COUNT; i++)
		scene.AddSphere(spheres_scene[i]);
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	window.RegisterTexture(scene.GetTexture());

	

	DevSphere spheres[SPHERE_COUNT];
	for (int i = 0; i < SPHERE_COUNT; i++) {

		spheres[i].position[0] = scene.spheres[i].position.x;
		spheres[i].position[1] = scene.spheres[i].position.y;
		spheres[i].position[2] = scene.spheres[i].position.z;

		spheres[i].radius = scene.spheres[i].radius;

		spheres[i].color[0] = scene.spheres[i].color.r;
		spheres[i].color[1] = scene.spheres[i].color.g;
		spheres[i].color[2] = scene.spheres[i].color.b;
	}


	Node nodeArr[2*SPHERE_COUNT-1];
	for (int i = 0; i < SPHERE_COUNT; i++) {
		nodeArr[i+SPHERE_COUNT-1] = Node{ -1, -1, spheres_scene[i].position.x, spheres_scene[i].position.y, spheres_scene[i].position.z, spheres_scene[i].radius, 0};
	}
	/*int row = 0;
	int col = 0;
	for (int i = 0; i < SPHERE_COUNT - 1; i++)
	{
		
		nodeArr[i] = Node{i+ (int)pow(2,row)+col, i+(int)pow(2,row) + col+1, 0, 0, 0, 0, 2};
		if (col == (int)pow(2, row)-1)
		{
			row++;
			col = 0;
		}
		else
		{
			col++;
		}
	}*/

	nodeArr[0] = Node{ 1,10,0,0,0,0,0 };
	nodeArr[1] = Node{ 2,9,0,0,0,0,1 };
	nodeArr[2] = Node{ 3,4,0,0,0,0,2 };
	nodeArr[3] = Node{ 6,5,0,0,0,0,2 };
	nodeArr[4] = Node{ 8,7,0,0,0,0,2 };


	//copy sphere and texture to gpu
	DevSphere* dev_spheres;
	unsigned char* dev_texture_data;
	float* dev_projection;
	float* dev_view;
	float* dev_camera_position;
	float* dev_light_postion;
	Node* dev_tree;

	cudaMalloc(&dev_tree, NODE_COUNT * sizeof(Node));
	cudaMalloc(&dev_projection, 16 * sizeof(float));
	cudaMalloc(&dev_view, 16 * sizeof(float));
	cudaMalloc(&dev_camera_position, 3 * sizeof(float));
	cudaMalloc(&dev_light_postion, 3 * sizeof(float));
	cudaMalloc(&dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char));
	cudaMalloc(&dev_spheres, SPHERE_COUNT * sizeof(DevSphere));

	cudaMemcpy(dev_tree, nodeArr, NODE_COUNT * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_texture_data, scene.GetTexture().data.data(), TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spheres, spheres, SPHERE_COUNT * sizeof(DevSphere), cudaMemcpyHostToDevice);

	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {

		float dt = GetTimePassed(last);

		window.ProccessInput(scene, dt);

		float r = 100000.0f;
		//scene.SetLight(Light(vec3(r * cos(scene.angle), 0, r * sin(scene.angle)), vec3(1, 1, 1)));
		//scene.SetLight(Light(vec3(r * cos(glfwGetTime()), 0, r * sin(glfwGetTime())), vec3(1, 1, 1)));
		scene.SetLight(Light(scene.GetCamera().position, vec3(1, 1, 1)));


		scene.UpdateTextureGpu(dev_texture_data, dev_spheres, dev_projection, dev_view, dev_camera_position, dev_light_postion, SPHERE_COUNT, dev_tree);
		//scene.UpdateTextureCpu(tree);

		// copy texture to cpu
		cudaMemcpy(scene.GetTexture().data.data(), dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.ClearScreen();
		window.Render(scene);
	}

	/*cudaFree(dev_spheres);
	cudaFree(dev_texture_data);
	cudaFree(dev_projection);
	cudaFree(dev_view);
	cudaFree(dev_camera_position);
	cudaFree(dev_light_postion);*/

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
	if (camera.fov > 90)
		camera.fov = 90;
	if (camera.fov < 1)
		camera.fov = 1;
}