cmake_minimum_required(VERSION 3.9)	

project(textureRender)
get_filename_component(ROOT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm REQUIRED)


# global headers
include_directories(includes)
file(GLOB_RECURSE textureRender_Headers 
    "includes/*.h" 
    "includes/*.cpp"
)

add_executable(textureRender face.cpp)
target_sources(textureRender 
PUBLIC
    ${textureRender_Headers}
)
	

target_include_directories(textureRender 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${textureRender_Headers}
)

target_link_libraries(textureRender glm::glm)
target_link_libraries(textureRender glad::glad)
target_link_libraries(textureRender glfw)

