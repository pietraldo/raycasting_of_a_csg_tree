
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
#include "Constants.h"

// Function declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
int createWindow(GLFWwindow*& window);
void InitImGui(GLFWwindow* window);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);



float lastX = TEXTURE_WIDHT / 2, lastY = TEXTURE_HEIGHT / 2;

Texture CreateTexture() {

	Texture texture;
	texture.channels = 3;
	texture.width = TEXTURE_WIDHT;
	texture.height = TEXTURE_HEIGHT;
	texture.data = std::vector<unsigned char>(TEXTURE_WIDHT * TEXTURE_HEIGHT * texture.channels, 200);

	return texture;
}

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




// Main function
int main() {

	GLFWwindow* window;
	if (createWindow(window) == -1) {
		return -1;
	}
	InitImGui(window);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	Texture texture = CreateTexture();
	RegisterTexture(texture);

	unsigned char* dev_texture_data;
	cudaMalloc(&dev_texture_data, sizeof(unsigned char) * texture.data.size());
	cudaMemcpy(dev_texture_data, texture.data.data(), sizeof(unsigned char) * texture.data.size(), cudaMemcpyHostToDevice);


	auto last = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {

		processInput(window);

		UpdateTextureCpu(texture);

		//clear the screen
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// generate texture in cuda
		//UpdateTextureOnGPU(dev_texture_data);
		//cudaMemcpy(texture.data.data(), dev_texture_data, sizeof(unsigned char) * texture.data.size(), cudaMemcpyDeviceToHost);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, texture.id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, texture.data.data());


		// Start ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Render ImGui window
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImVec2 m_WindowSize = ImVec2(TEXTURE_WIDHT, TEXTURE_HEIGHT);
		ImGui::SetNextWindowSize(m_WindowSize);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.0f, 0.0f });
		ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		ImVec2 m_ViewportSize = ImGui::GetContentRegionAvail();
		ImGui::Image((ImTextureID)(uintptr_t)texture.id, m_ViewportSize, { 0, 1 }, { 1, 0 });
		ImGui::End();
		ImGui::PopStyleVar();

		ImGui::Begin("Debug Info");
		ImGui::Text("Camera Position: %f %f %f", camera.position.x, camera.position.y, camera.position.z);
		ImGui::Text("Camera Direction: %f %f %f", camera.direction.x, camera.direction.y, camera.direction.z);
		ImGui::Text("Camera Yaw: %f", camera.yaw);
		ImGui::Text("Camera Pitch: %f", camera.pitch);
		ImGui::End();

		// Render ImGui data
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers and poll event
		glfwSwapBuffers(window);
		glfwPollEvents();

		auto time = glfwGetTime();
		//std::cout << 1 / (time - last) << std::endl;
		last = time;
	}





	return 0;
}


// Function definitions
bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	/*if (GLFW_PRESS != glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT))
		return;
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	camera.yaw += xoffset;
	camera.pitch += yoffset;

	if (camera.pitch > 89.0f)
		camera.pitch = 89.0f;
	if (camera.pitch < -89.0f)
		camera.pitch = -89.0f;

	glm::vec3 direction;
	direction.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
	direction.y = sin(glm::radians(camera.pitch));
	direction.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
	camera.direction = glm::normalize(direction);

	cout << xpos << " " << ypos << endl;*/
	float dx = xpos - lastX;
	float dy = ypos - lastY;
	lastY = ypos;
	lastX = xpos;
	angleX -= dy * 0.1f;
	angleY -= dx * 0.1f;
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.position += camera.direction*0.1f;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.position -= camera.direction*0.1f;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.position -= glm::normalize(glm::cross(camera.direction, camera.up))*0.1f;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.position += glm::normalize(glm::cross(camera.direction, camera.up))*0.1f;
	
}

// Callback for window resizing
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

// Create a GLFW window and set up OpenGL context
int createWindow(GLFWwindow*& window) {
	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return -1;
	}

	// Configure GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Texture Viewer", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Load OpenGL functions using GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		return -1;
	}

	return 0;
}


// Initialize ImGui
void InitImGui(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	// Initialize backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

