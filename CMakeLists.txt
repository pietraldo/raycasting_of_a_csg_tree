cmake_minimum_required(VERSION 3.25)

project(raycasting_of_a_csg_tree LANGUAGES CUDA CXX C)

add_subdirectory(raycasting_of_a_csg_tree)
add_subdirectory(externals/glfw)
add_subdirectory(externals/glad)
add_subdirectory(externals/imgui)
add_subdirectory(externals/glm)

