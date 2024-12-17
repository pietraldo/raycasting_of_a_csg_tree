# CMakeList.txt: projekt CMake dla elementu raycasting_of_a_csg_tree, dołącz źródło i zdefiniuj
# tutaj logikę specyficzną dla projektu.
#

# Dodaj źródło do pliku wykonywalnego tego projektu.
add_executable (raycasting_of_a_csg_tree "raycasting_of_a_csg_tree.cpp" "raycasting_of_a_csg_tree.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET raycasting_of_a_csg_tree PROPERTY CXX_STANDARD 20)
endif()

# TODO: Dodaj testy i zainstaluj elementy docelowe w razie potrzeby.
