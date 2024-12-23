
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

	GLFWwindow* window;
	if (createWindow(window) == -1) {
		return -1;
	}
	InitImGui(window);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


	scene = Scene();
	scene.SetCamera(Camera(vec3(0, 0, 0)));
	scene.AddSphere(Sphere(vec3(0, 0, 7), 1.0f));
	scene.AddSphere(Sphere(vec3(0, 0, 2), 0.5f));
	scene.SetLight(Light(vec3(0, 0, -4), vec3(1, 1, 1)));

	RegisterTexture(scene.GetTexture());

	

	auto last = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		
		auto time = glfwGetTime();
		float dt = time - last;
		std::cout << 1 / dt << std::endl;

		last = time;
		processInput(window,dt);
		
		float r = 10.0f;
		scene.SetLight(Light(vec3(r * cos(time), 0, r * sin(time)), vec3(1,1,1)));

		scene.UpdateTextureCpu();

		//clear the screen
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// copy to opengl
		glBindTexture(GL_TEXTURE_2D, scene.GetTexture().id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDHT, TEXTURE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, scene.GetTexture().data.data());

		renderWindow(window, scene);
		

		// Swap buffers and poll event
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}

void renderWindow(GLFWwindow* window, Scene& scene)
{
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
	ImGui::Image((ImTextureID)(uintptr_t)scene.GetTexture().id, m_ViewportSize, {0, 1}, {1, 0});
	ImGui::End();
	ImGui::PopStyleVar();

	ImGui::Begin("Debug Info");
	ImGui::Text("Camera Position: %f %f %f", scene.GetCamera().position.x, scene.GetCamera().position.y, scene.GetCamera().position.z);
	ImGui::Text("Camera Direction: %f %f %f", scene.GetCamera().direction.x, scene.GetCamera().direction.y, scene.GetCamera().direction.z);
	ImGui::Text("Camera Yaw: %f", scene.GetCamera().yaw);
	ImGui::Text("Camera Pitch: %f", scene.GetCamera().pitch);
	ImGui::End();


	// Render ImGui data
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


// Function definitions
bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (GLFW_PRESS != glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT))
	{
		lastX = xpos;
		lastY = ypos;
		return;
	}
		
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

	Camera& camera = scene.GetCamera();

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
}
float speed = 2.0f;
void processInput(GLFWwindow* window, float timePassed)
{
	Camera& camera = scene.GetCamera();
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.position += camera.direction*speed* timePassed;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.position -= camera.direction * speed * timePassed;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.position -= glm::normalize(glm::cross(camera.direction, camera.up)) * speed * timePassed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.position += glm::normalize(glm::cross(camera.direction, camera.up)) * speed * timePassed;
	
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

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	Camera& camera = scene.GetCamera();
	camera.fov -= (float)yoffset;
	if(camera.fov>90)
		camera.fov = 90;
	if (camera.fov < 1)
		camera.fov = 1;
}