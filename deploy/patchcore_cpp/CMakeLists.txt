cmake_minimum_required(VERSION 3.18)
project(patchcore_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
#   - TensorRT 10.11: pass -DTRT_DIR="D:/TensorRT-10.11.0.33" (root of the zip
#     extraction; must contain include/ and lib/)
#   - CUDA Toolkit matching your TensorRT build (12.9 or 13.0 for TRT 10.11)
#   - OpenCV 4.x: pass -DOpenCV_DIR="..." if not found automatically
# ---------------------------------------------------------------------------
set(TRT_DIR "" CACHE PATH "TensorRT installation root")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT TRT_DIR)
    if(DEFINED ENV{TRT_DIR})
        set(TRT_DIR $ENV{TRT_DIR})
    else()
        message(FATAL_ERROR "Set -DTRT_DIR=<TensorRT root> or the TRT_DIR environment variable")
    endif()
endif()

# TRT 10 on Windows ships versioned libs (nvinfer_10.lib); Linux: libnvinfer.so.10
find_library(NVINFER_LIB NAMES nvinfer_10 nvinfer PATHS "${TRT_DIR}/lib" REQUIRED NO_DEFAULT_PATH)

add_executable(patchcore_trt main.cpp)
target_include_directories(patchcore_trt PRIVATE "${TRT_DIR}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(patchcore_trt PRIVATE ${NVINFER_LIB} CUDA::cudart ${OpenCV_LIBS})

if(WIN32)
    # Copy the TRT runtime DLL next to the executable for convenience.
    add_custom_command(TARGET patchcore_trt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TRT_DIR}/lib/nvinfer_10.dll" "$<TARGET_FILE_DIR:patchcore_trt>")
endif()
