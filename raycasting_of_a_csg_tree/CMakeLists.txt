﻿add_executable (raycasting_of_a_csg_tree main.cpp kernels.cuh kernels.cu Camera.cpp Camera.h Constants.h  Light.cpp Light.h Scene.cpp scene.h Texture.cpp Texture.h Tree.cpp Tree.h Window.cpp Window.h TreeParser.cpp TreeParser.h GPUdata.h Test.cpp Test.h)


target_include_directories(raycasting_of_a_csg_tree PRIVATE ${CMAKE_SOURCE_DIR}/raycasting_of_a_csg_tree)
target_include_directories(raycasting_of_a_csg_tree PRIVATE ${CMAKE_SOURCE_DIR}/externals/imgui/imgui/backends)
target_link_libraries(raycasting_of_a_csg_tree glfw glad imgui glm)


