cmake_minimum_required(VERSION 3.16)
project(canned_food_crack_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
# If OpenCV_DIR is not set, default to standard local path C:/opencv/build
if(NOT OpenCV_DIR AND EXISTS "C:/opencv/build")
    set(OpenCV_DIR "C:/opencv/build")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} src)

add_executable(can_crack_inspector
    src/main.cpp
)

target_link_libraries(can_crack_inspector PRIVATE ${OpenCV_LIBS})

# Set runtime output directory
set_target_properties(can_crack_inspector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
