
cmake_minimum_required(VERSION 3.16.3)
project(nl-means)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(
     CUDA_NVCC_FLAGS
     ${CUDA_NVCC_FLAGS};
     -O3 -lineinfo
     -gencode=arch=compute_89,code=sm_89
     )

include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(
     nl-means
     nlm.cpp main.cu main.h cuda_utility.cpp cuda_utility.h)

target_link_libraries( nl-means ${OpenCV_LIBS} )


