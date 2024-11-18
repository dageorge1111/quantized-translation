cmake_minimum_required(VERSION 3.10)
project(ONNXTranslation)
include_directories(/usr/local/include/eigen3)
set(CMAKE_CXX_STANDARD 14)

find_package(onnxruntime REQUIRED)

add_executable(ONNXTranslation main.cpp)
target_link_libraries(ONNXTranslation PRIVATE onnxruntime)

