cmake_minimum_required(VERSION 3.26)
project(fish_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

set(PYBIND11_FINDPYTHON ON)
execute_process(
    COMMAND python3 -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Python_EXECUTABLE ${PYTHON_EXECUTABLE_PATH})
# find_package(pybind11 REQUIRED)
add_subdirectory(extern/pybind11)

include_directories(${CMAKE_SOURCE_DIR}/src/Eigen)
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
add_library(fish_sim_objs OBJECT ${CPP_SOURCES})
target_compile_options(fish_sim_objs PRIVATE -O2 -march=native -flto -funroll-loops -DNDEBUG -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)

pybind11_add_module(fish_sim bindings.cpp)
target_include_directories(fish_sim PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(fish_sim PRIVATE fish_sim_objs "-framework Accelerate")
install(TARGETS fish_sim DESTINATION .)
