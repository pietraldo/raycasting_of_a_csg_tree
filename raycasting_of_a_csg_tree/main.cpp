
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

// Function declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
int createWindow(GLFWwindow*& window);
void InitImGui(GLFWwindow* window);
void processInput(GLFWwindow* window);


// Screen dimensions
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;





Texture CreateTexture() {

    Texture texture;
    texture.channels = 3;
    texture.width = SCR_WIDTH;
    texture.height = SCR_HEIGHT;
    texture.data = std::vector<unsigned char>(SCR_WIDTH * SCR_HEIGHT * texture.channels, 200);

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

    Texture texture = CreateTexture();
    RegisterTexture(texture);

    unsigned char* dev_texture_data;
    cudaMalloc(&dev_texture_data, sizeof(unsigned char) * texture.data.size());
    cudaMemcpy(dev_texture_data, texture.data.data(), sizeof(unsigned char) * texture.data.size(), cudaMemcpyHostToDevice);


    auto last = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {

		processInput(window);

        UpdateTextureCpu(texture);

        // generate texture in cuda
        //UpdateTextureOnGPU(dev_texture_data);
        //cudaMemcpy(texture.data.data(), dev_texture_data, sizeof(unsigned char) * texture.data.size(), cudaMemcpyDeviceToHost);

        // copy to opengl
        glBindTexture(GL_TEXTURE_2D, texture.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, texture.data.data());


        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render ImGui window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImVec2 m_WindowSize = ImVec2(SCR_WIDTH, SCR_HEIGHT);
        ImGui::SetNextWindowSize(m_WindowSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.0f, 0.0f });
        ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImVec2 m_ViewportSize = ImGui::GetContentRegionAvail();
        ImGui::Image((ImTextureID)(uintptr_t)texture.id, m_ViewportSize, { 0, 1 }, { 1, 0 });
        ImGui::End();
        ImGui::PopStyleVar();

        // Render ImGui data
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll event
        glfwSwapBuffers(window);
        glfwPollEvents();

        auto time = glfwGetTime();
        std::cout << 1 / (time - last) << std::endl;
        last = time;
    }





    return 0;
}


// Function definitions

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.z += 0.1f;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.z -= 0.1f;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.x -= 0.1f;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.x += 0.1f;
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

