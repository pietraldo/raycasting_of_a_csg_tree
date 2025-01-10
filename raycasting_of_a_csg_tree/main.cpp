
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


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
float GetTimePassed(float& last);


#include "kernels.cuh"

Scene scene;

void CreateParts(int* part,vector<Node>& tree, int node, bool isLeft, const int SphereCount)
{
	int ll, lr, rl, rr;
	if (tree[tree[node].left].left == -1) // is leaf
	{
		ll = (tree[node].left - SphereCount+1)*2;
		lr = (tree[node].left - SphereCount+1)*2+1;
	}
	else
	{
		CreateParts(part, tree, tree[node].left, true, SphereCount);
		ll = part[tree[node].left * 4];
		lr = part[tree[node].left * 4 + 3];
	}
	if (tree[tree[node].right].left == -1) // is leaf
	{
		rl = (tree[node].right - SphereCount + 1) * 2;
		rr = (tree[node].right - SphereCount + 1) * 2 + 1;
	}
	else
	{
		CreateParts(part, tree, tree[node].right, false, SphereCount);
		rl = part[tree[node].right * 4];
		rr = part[tree[node].right * 4 + 3];
	}
	


	part[node * 4] = ll;
	part[node * 4 + 1] = lr;
	part[node * 4 + 2] = rl;
	part[node * 4 + 3] = rr;
}

// Main function
int main() {

	

	Window window(SCR_WIDTH, SCR_HEIGHT);

	glfwSetScrollCallback(window.GetWindow(), scroll_callback);


	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, -8)));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	window.RegisterTexture(scene.GetTexture());


	
	TreeParser parser("C:/Users/pietr/Documents/studia/karty graficzne/csg_model1.txt");
	parser.Parse();
	
	int SPHERE_COUNT = parser.num_spheres;
	int CUBES_COUNT = parser.num_cubes;
	int CYLINDER_COUNT = parser.num_cylinders;
	int SHAPE_COUNT = SPHERE_COUNT + CUBES_COUNT+CYLINDER_COUNT;
	int NODE_COUNT = 2 * SHAPE_COUNT - 1;


	int* parts= new int[4 * (SHAPE_COUNT - 1)];
	CreateParts(parts, parser.nodes, 0, true, SHAPE_COUNT);


	//copy sphere and texture to gpu
	unsigned char* dev_texture_data;
	float* dev_projection;
	float* dev_view;
	float* dev_camera_position;
	float* dev_light_postion;
	Node* dev_tree;
	int* dev_parts;
	Sphere* dev_spheres;
	Cube* dev_cubes;
	Cylinder* dev_cylinders;

	float* dev_intersecion_points;
	float* dev_intersection_result;

	Sphere* spheres = parser.spheres.data();
	Cube* cubes = parser.cubes.data();
	Cylinder* cylinders = parser.cylinders.data();

	cudaError_t err;

	err = cudaMalloc(&dev_spheres, MAX_SHAPES * sizeof(Sphere));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_spheres error: %s\n", cudaGetErrorString(err));
	}

	err = cudaMalloc(&dev_cubes, MAX_SHAPES * sizeof(Cube));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_cubes error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_cylinders, MAX_SHAPES * sizeof(Cylinder));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_cylinders error: %s\n", cudaGetErrorString(err));
	}

	parser.AttachShapes(dev_cubes, dev_spheres, dev_cylinders);

	err = cudaMalloc(&dev_tree, NODE_COUNT * sizeof(Node));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_projection, 16 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_view, 16 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_camera_position, 3 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_light_postion, 3 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_intersecion_points, TEXTURE_WIDHT * TEXTURE_HEIGHT * SHAPE_COUNT * 2 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_intersection_result, TEXTURE_WIDHT * TEXTURE_HEIGHT * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_parts, 4*(SHAPE_COUNT - 1) * sizeof(int));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}


	cudaMemcpy(dev_tree, parser.nodes.data(), NODE_COUNT * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_texture_data, scene.GetTexture().data.data(), TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_parts, parts, 4*(SHAPE_COUNT - 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spheres, spheres, MAX_SHAPES * sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cubes, cubes, MAX_SHAPES * sizeof(Cube), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cylinders, cylinders, MAX_SHAPES * sizeof(Cylinder), cudaMemcpyHostToDevice);

	int cam_direction = 1;
	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {

		float dt = GetTimePassed(last);

		window.ProccessInput(scene, dt);

		float r = 100000.0f;

		if (scene.GetLight().rotateLight)
		{
			scene.SetLightPosition(vec3(r * cos(glfwGetTime()), 0, r * sin(glfwGetTime())));
		}
		else
		{
			scene.SetLightPosition(vec3(r* cos(scene.angle), 0, r* sin(scene.angle)));
		}
		
		
		if (scene.GetCamera().animation)
		{
			scene.GetCamera().yaw += 0.25;
			scene.GetCamera().pitch += cam_direction * 0.05;
			if (scene.GetCamera().pitch > 20)
			{
				cam_direction = -1;
			}
			if (scene.GetCamera().pitch < -20)
			{
				cam_direction = 1;
			}
		}
		//scene.SetLight(Light(scene.GetCamera().position, vec3(1, 1, 1)));


		scene.UpdateTextureGpu(dev_texture_data, dev_projection, dev_view, dev_camera_position, dev_light_postion,
			SHAPE_COUNT, dev_tree, dev_intersecion_points, dev_intersection_result, dev_parts, dev_spheres, dev_cubes);
		

		// copy texture to cpu
		cudaMemcpy(scene.GetTexture().data.data(), dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.ClearScreen();
		window.Render(scene);

		
	}

	cudaFree(dev_spheres);
	cudaFree(dev_texture_data);
	cudaFree(dev_projection);
	cudaFree(dev_view);
	cudaFree(dev_camera_position);
	cudaFree(dev_light_postion);

	return 0;
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