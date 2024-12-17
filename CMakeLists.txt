cmake_minimum_required(VERSION 3.25)



project(Raycasting LANGUAGES CUDA CXX C)

add_subdirectory(raycasting_of_a_csg_tree)
add_subdirectory(externals/glfw)
add_subdirectory(externals/glad)
