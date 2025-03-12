#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Scene.h"
#include "Camera.h"

class Window
{
private:
	GLFWwindow* window;
	float lastMouseX = 0, lastMouseY = 0;

	void ProccessKeys(Scene& scene, float dt);
	void ProccessMouse(Scene& scene, float dt);

public:
	Window(const int width, const int height);
	void Render(Scene& scene, float dt);
	void ProccessInput(Scene& scene, float dt);
	bool ShouldCloseWindow();
	GLFWwindow* GetWindow() { return window; }
	void ClearScreen();
	void RegisterTexture(Texture& texture);
};

