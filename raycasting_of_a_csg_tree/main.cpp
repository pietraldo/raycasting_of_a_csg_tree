
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

	Sphere sphere1(vec3(0, 0, 2), 1.0f);
	Sphere sphere2(vec3(0.5, 0, 2), 1.0f);
	Sphere sphere3(vec3(0, 0.7, 2), 1.0f);

	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, 0)));
	scene.AddSphere(sphere1);
	scene.AddSphere(sphere2);
	scene.AddSphere(sphere3);
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));


	//build tree
	Tree tree;
	Node* one = new Node{
		true,
		nullptr,
		nullptr,
		&sphere1,
		nullptr
	}
	;
	Node* two = new Node{
		true,
		nullptr,
		nullptr,
		&sphere2,
		nullptr
	};
	Node* three = new Node{
		true,
		nullptr,
		nullptr,
		&sphere3,
		nullptr
	};
	Node* add = new Node{
		false,
		one,
		two,
		nullptr,
		Tree::Subtraction
	};
	Node* root = new Node{
		false,
		add,
		three,
		nullptr,
		Tree::Subtraction
	};
	tree.SetRoot(root);


	window.RegisterTexture(scene.GetTexture());


	//DevSphere spheres[8];
	//for (int i = 0; i < 8; i++) {

	//	spheres[i].position[0] = scene.spheres[i].position.x;
	//	spheres[i].position[1] = scene.spheres[i].position.y;
	//	spheres[i].position[2] = scene.spheres[i].position.z;

	//	spheres[i].radius = scene.spheres[i].radius;

	//	spheres[i].color[0] = scene.spheres[i].color.r;
	//	spheres[i].color[1] = scene.spheres[i].color.g;
	//	spheres[i].color[2] = scene.spheres[i].color.b;
	//}

	////copy sphere and texture to gpu
	//DevSphere* dev_spheres;
	//unsigned char* dev_texture_data;
	//float* dev_projection;
	//float* dev_view;
	//float* dev_camera_position;
	//float* dev_light_postion;


	//cudaMalloc(&dev_projection, 16 * sizeof(float));
	//cudaMalloc(&dev_view, 16 * sizeof(float));
	//cudaMalloc(&dev_camera_position, 3 * sizeof(float));
	//cudaMalloc(&dev_light_postion, 3 * sizeof(float));
	//cudaMalloc(&dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char));
	//cudaMalloc(&dev_spheres, 8 * sizeof(DevSphere));


	//cudaMemcpy(dev_texture_data, scene.GetTexture().data.data(), TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_spheres, spheres, 8 * sizeof(DevSphere), cudaMemcpyHostToDevice);

	float last = glfwGetTime();
	while (!window.ShouldCloseWindow()) {

		float dt = GetTimePassed(last);

		window.ProccessInput(scene, dt);

		float r = 10.0f;
		//scene.SetLight(Light(vec3(r * cos(glfwGetTime()), 0, r * sin(glfwGetTime())), vec3(1, 1, 1)));
		scene.SetLight(Light(scene.GetCamera().position, vec3(1, 1, 1)));


		//scene.UpdateTextureGpu(dev_texture_data, dev_spheres, dev_projection, dev_view, dev_camera_position, dev_light_postion, 8);
		scene.UpdateTextureCpu(tree);

		// copy texture to cpu
		//cudaMemcpy(scene.GetTexture().data.data(), dev_texture_data, TEXTURE_WIDHT * TEXTURE_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

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