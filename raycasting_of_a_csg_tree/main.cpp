
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

void CreateParts(int* part,Node* tree, int node, bool isLeft, const int SphereCount)
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

	const int SPHERE_COUNT = 16;
	const int NODE_COUNT = 2 * SPHERE_COUNT - 1;

	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, -8)));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	window.RegisterTexture(scene.GetTexture());


	Node nodeArr[2 * SPHERE_COUNT - 1];

	for (int i = SPHERE_COUNT - 1; i < 2 * SPHERE_COUNT - 1; i++)
	{
		float x = rand()/(float)RAND_MAX*4;
		float y = rand()/(float)RAND_MAX*4;
		float z = rand()/(float)RAND_MAX*4;
		float r = 0.5f;
		int parent = (i - 1) / 2;
		nodeArr[i] = Node{ -1,-1,parent,x,y,z,r,0};
	}
	for (int i = 0; i < SPHERE_COUNT - 1; i++)
	{
		int left = 2 * i + 1;
		int right = 2 * i + 2;
		int parent = (i - 1) / 2;
		nodeArr[i] = Node{left, right, parent, 0,0,0,0,2};
	}
	nodeArr[0].parent = -1;
	nodeArr[0].operation = 2;
	/*nodeArr[0] = Node{ 1,10,-1,0,0,0,0,0 };
	nodeArr[1] = Node{ 2,9,0,0,0,0,0,1 };
	nodeArr[2] = Node{ 3,4,1,0,0,0,0,2 };
	nodeArr[3] = Node{ 5,6,2,0,0,0,0,2 };
	nodeArr[4] = Node{ 7,8,2,0,0,0,0,2 };

	nodeArr[5] = Node{ -1,-1,3, 1,0,0,1,0 };
	nodeArr[6] = Node{ -1,-1,3, -1,0,0,1,0 };
	nodeArr[7] = Node{ -1,-1,4, 0,1,0,1,0 };
	nodeArr[8] = Node{ -1,-1,4, 0,-1,0,1,0 };
	nodeArr[9] = Node{ -1,-1,1, 0,0,0,1.5,0 };
	nodeArr[10] = Node{ -1,-1,0, 0,0,0,0.5,0 };*/

	int parts[4 * (SPHERE_COUNT - 1)];
	CreateParts(parts, nodeArr, 0, true, SPHERE_COUNT);

	//copy sphere and texture to gpu
	unsigned char* dev_texture_data;
	float* dev_projection;
	float* dev_view;
	float* dev_camera_position;
	float* dev_light_postion;
	Node* dev_tree;
	int* dev_parts;

	float* dev_intersecion_points;
	float* dev_intersection_result;

	cudaError_t err;
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
	err = cudaMalloc(&dev_intersecion_points, TEXTURE_WIDHT * TEXTURE_HEIGHT * SPHERE_COUNT * 2 * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_intersection_result, TEXTURE_WIDHT * TEXTURE_HEIGHT * sizeof(float));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc(&dev_parts, 4*(SPHERE_COUNT - 1) * sizeof(int));
	if (err != cudaSuccess) {
		printf("cudaMalloc dev_tree error: %s\n", cudaGetErrorString(err));
	}


	cudaMemcpy(dev_tree, nodeArr, NODE_COUNT * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_texture_data, scene.GetTexture().data.data(), TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_parts, parts, 4*(SPHERE_COUNT - 1) * sizeof(int), cudaMemcpyHostToDevice);

	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {

		float dt = GetTimePassed(last);

		window.ProccessInput(scene, dt);

		float r = 100000.0f;
		scene.SetLight(Light(vec3(r * cos(scene.angle), 0, r * sin(scene.angle)), vec3(1, 1, 1)));
		//scene.SetLight(Light(vec3(r * cos(glfwGetTime()), 0, r * sin(glfwGetTime())), vec3(1, 1, 1)));
		//scene.SetLight(Light(scene.GetCamera().position, vec3(1, 1, 1)));


		scene.UpdateTextureGpu(dev_texture_data, dev_projection, dev_view, dev_camera_position, dev_light_postion,
			SPHERE_COUNT, dev_tree, dev_intersecion_points, dev_intersection_result, dev_parts);
		//scene.UpdateTextureCpu(tree);

		// copy texture to cpu
		cudaMemcpy(scene.GetTexture().data.data(), dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		window.ClearScreen();
		window.Render(scene);

		//_sleep(100000);
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