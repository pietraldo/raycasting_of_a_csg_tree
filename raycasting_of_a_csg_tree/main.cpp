
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
#include "TreeParser.h"
#include "kernels.cuh"
#include "GPUdata.h"

float GetTimePassed(float& last);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

GPUdata MallocCopyDataToGPU(TreeParser parser);
void FreeMemory(GPUdata data);

Scene scene;
int main() {

	TreeParser parser("C:/Users/pietr/Documents/studia/karty graficzne/csg_model1.txt");
	if (parser.Parse())
	{
		cout << "Parsing successful" << endl;
	}
	else
	{
		cout << "Parsing failed" << endl;
		return -1;
	}

	Window window(SCR_WIDTH, SCR_HEIGHT);

	scene=Scene();
	scene.SetCamera(Camera(vec3(0, 0, -8)));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	window.RegisterTexture(scene.GetTexture());
	
	glfwSetScrollCallback(window.GetWindow(), scroll_callback);


	GPUdata gpuData = MallocCopyDataToGPU(parser);

	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {

		float dt = GetTimePassed(last);

		window.ProccessInput(scene, dt);

		scene.Update();

		scene.UpdateTextureGpu(gpuData);
		

		// copy texture to cpu
		cudaMemcpy(scene.GetTexture().data.data(), gpuData.dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.ClearScreen();
		window.Render(scene);

		
	}

	FreeMemory(gpuData);

	return 0;
}


GPUdata MallocCopyDataToGPU(TreeParser parser)
{
	GPUdata data;

	int SPHERE_COUNT = parser.num_spheres;
	int CUBES_COUNT = parser.num_cubes;
	int CYLINDER_COUNT = parser.num_cylinders;
	int SHAPE_COUNT = SPHERE_COUNT + CUBES_COUNT + CYLINDER_COUNT;
	int NODE_COUNT = 2 * SHAPE_COUNT - 1;

	data.ShapeCount = SHAPE_COUNT;

	Sphere* spheres = parser.spheres.data();
	Cube* cubes = parser.cubes.data();
	Cylinder* cylinders = parser.cylinders.data();

	cudaError_t err;

	err = cudaMalloc(&data.dev_spheres, MAX_SHAPES * sizeof(Sphere));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_spheres error: %s\n", cudaGetErrorString(err));
	}

	err = cudaMalloc(&data.dev_cubes, MAX_SHAPES * sizeof(Cube));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_cubes error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_cylinders, MAX_SHAPES * sizeof(Cylinder));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_cylinders error: %s\n", cudaGetErrorString(err));
	}

	parser.AttachShapes(data.dev_cubes, data.dev_spheres, data.dev_cylinders);

	err = cudaMalloc(&data.dev_tree, NODE_COUNT * sizeof(Node));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_projection, 16 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_view, 16 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_camera_position, 3 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_light_postion, 3 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}

	err = cudaMalloc(&data.dev_intersection_result, TEXTURE_WIDHT * TEXTURE_HEIGHT * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&data.dev_parts, 4 * (SHAPE_COUNT - 1) * sizeof(int));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}

	printf("parser parts %d\n", parser.parts[0]);

	cudaMemcpy(data.dev_tree, parser.nodes.data(), NODE_COUNT * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_texture_data, scene.GetTexture().data.data(), TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_parts, parser.parts, 4 * (SHAPE_COUNT - 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_spheres, spheres, MAX_SHAPES * sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_cubes, cubes, MAX_SHAPES * sizeof(Cube), cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_cylinders, cylinders, MAX_SHAPES * sizeof(Cylinder), cudaMemcpyHostToDevice);

	return data;
}

void FreeMemory(GPUdata data)
{
	cudaFree(data.dev_spheres);
	cudaFree(data.dev_texture_data);
	cudaFree(data.dev_projection);
	cudaFree(data.dev_view);
	cudaFree(data.dev_camera_position);
	cudaFree(data.dev_light_postion);
	cudaFree(data.dev_tree);
	cudaFree(data.dev_intersection_result);
	cudaFree(data.dev_parts);
	cudaFree(data.dev_cubes);
	cudaFree(data.dev_cylinders);
}

float GetTimePassed(float& last) {
	auto time = glfwGetTime();
	float dt = time - last;
	last = time;
	//std::cout << 1 / dt << std::endl;
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