add_executable (raycasting_of_a_csg_tree "main.cpp")

target_include_directories(raycasting_of_a_csg_tree PRIVATE ${CMAKE_SOURCE_DIR}/raycasting_of_a_csg_tree)
target_include_directories(raycasting_of_a_csg_tree PRIVATE ${CMAKE_SOURCE_DIR}/externals/imgui/imgui/backends)
target_link_libraries(raycasting_of_a_csg_tree glfw glad imgui)

