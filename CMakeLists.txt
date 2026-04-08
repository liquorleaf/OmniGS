cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(omnigs LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

find_package(realsense2)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    third_party/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR})

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
set_target_properties(simple_knn PROPERTIES CUDA_ARCHITECTURES "120")
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "120")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp)
target_link_libraries(gaussian_mapper
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer)

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h)
target_link_libraries(gaussian_viewer
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################
add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result
    gaussian_viewer
    gaussian_mapper)

# A simple model to describe the camera model
add_executable(simple_cloud examples/simple_cloud.cpp)
target_link_libraries(simple_cloud
    gaussian_viewer
    gaussian_mapper)

# Train openMVG scenes
add_executable(train_openmvg_lonlat examples/train_openmvg_lonlat.cpp)
target_link_libraries(train_openmvg_lonlat
    gaussian_viewer
    gaussian_mapper)

# Test openMVG scenes
add_executable(test_openmvg_lonlat examples/test_openmvg_lonlat.cpp)
target_link_libraries(test_openmvg_lonlat
    gaussian_viewer
    gaussian_mapper)

# Train 360Roam scenes
add_executable(train_360roam examples/train_360roam.cpp)
target_link_libraries(train_360roam
    gaussian_viewer
    gaussian_mapper)

# Train EgoNeRF scenes
add_executable(train_egonerf_omniblender examples/train_egonerf_omniblender.cpp)
target_link_libraries(train_egonerf_omniblender
    gaussian_viewer
    gaussian_mapper)

add_executable(train_egonerf_ricoh360 examples/train_egonerf_ricoh360.cpp)
target_link_libraries(train_egonerf_ricoh360
    gaussian_viewer
    gaussian_mapper)
