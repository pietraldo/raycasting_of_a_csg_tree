#include "Window.h"

Window::Window(const int width, const int height)
{
	lastMouseX = width / 2;
	lastMouseY = height / 2;

	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
	}

	// Configure GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	window = glfwCreateWindow(width, height, "Texture Viewer", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Load OpenGL functions using GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	// Initialize backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void Window::ProccessInput(Scene& scene, float dt)
{
	ProccessKeys(scene, dt);
	ProccessMouse(scene, dt);
}

void Window::ProccessKeys(Scene& scene, float dt)
{
	Camera& camera = scene.GetCamera();
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		if(camera.rotateScene)
			camera.r -= camera.speed * dt;
		else
			camera.position += camera.direction * camera.speed * dt;
	}
		
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		if(camera.rotateScene)
			camera.r += camera.speed * dt;
		else
			camera.position -= camera.direction * camera.speed * dt;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.position -= glm::normalize(glm::cross(camera.direction, camera.up)) * camera.speed * dt;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.position += glm::normalize(glm::cross(camera.direction, camera.up)) * camera.speed * dt;
	camera.UpdatePosition();
}

void Window::ProccessMouse(Scene& scene, float dt)
{
	double mouseX, mouseY;
	glfwGetCursorPos(window, &mouseX, &mouseY);

	if (GLFW_PRESS != glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT))
	{
		lastMouseX = mouseX;
		lastMouseY = mouseY;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		return;
	}
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	float xoffset = mouseX - lastMouseX;
	float yoffset = lastMouseY- mouseY;
	lastMouseX = mouseX;
	lastMouseY = mouseY;

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

	camera.UpdatePosition();
	
}

void Window::Render(Scene& scene)
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
	ImGui::Image((ImTextureID)(uintptr_t)scene.GetTexture().id, m_ViewportSize, { 0, 1 }, { 1, 0 });
	ImGui::End();
	ImGui::PopStyleVar();

	ImGui::Begin("Debug Info");
	ImGui::Text("Camera Position: %f %f %f", scene.GetCamera().position.x, scene.GetCamera().position.y, scene.GetCamera().position.z);
	ImGui::Text("Camera Direction: %f %f %f", scene.GetCamera().direction.x, scene.GetCamera().direction.y, scene.GetCamera().direction.z);
	ImGui::Text("Camera Yaw: %f", scene.GetCamera().yaw);
	ImGui::Text("Camera Pitch: %f", scene.GetCamera().pitch);
	ImGui::Text("Fps: %f", ImGui::GetIO().Framerate);
	ImGui::SliderFloat("Angle", &scene.angle, 0, 2 * 3.14159265);
	ImGui::SliderFloat("Angle Right", &scene.GetCamera().yaw, 0, 360);
	ImGui::SliderFloat("Angle Up", &scene.GetCamera().pitch, 0, 360);
	ImGui::Checkbox("Rotate Scene", &scene.GetCamera().rotateScene);
	ImGui::End();


	// Render ImGui data
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Swap buffers and poll event
	glfwSwapBuffers(window);
	glfwPollEvents();
}

bool Window::ShouldCloseWindow()
{
	return glfwWindowShouldClose(window);
}

void Window::ClearScreen()
{
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}

void Window::RegisterTexture(Texture& texture)
{
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