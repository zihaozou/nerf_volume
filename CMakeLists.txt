cmake_minimum_required(VERSION 3.18)

PROJECT(
	nerf_volume
	LANGUAGES CXX CUDA
)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cutlass/CMakeLists.txt")
	message(FATAL_ERROR
		"Some tiny-cuda-nn dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)

# adapted from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#set(CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)

# Remove unsupported architectures
list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "PTX")

if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
endif()

# If the CUDA version does not permit targeting Ampere, don't do so.
if ((80 IN_LIST CMAKE_CUDA_ARCHITECTURES OR 86 IN_LIST CMAKE_CUDA_ARCHITECTURES) AND CUDA_VERSION VERSION_LESS 11.0)
	message(WARNING "CUDA version ${CUDA_VERSION} is too low for targeting Ampere GPUs. Reverting to compute capability 75.")
	list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES 80 86)
	if (NOT CMAKE_CUDA_ARCHITECTURES)
		list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
	endif()
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (TCNN_HAS_PARENT)
	set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
endif()

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND TCNN_DEFINITIONS -DTCNN_SHAMPOO)
endif()

add_definitions(${TCNN_DEFINITIONS})
if (TCNN_HAS_PARENT)
	set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
endif()

if (MSVC)
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-lafcuda")
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("debug mode")
	list(APPEND CUDA_NVCC_FLAGS "-g")
	list(APPEND CUDA_NVCC_FLAGS "-O0")
	list(APPEND CUDA_NVCC_FLAGS "-G")
else()
	message("release mode")
	list(APPEND CUDA_NVCC_FLAGS "-O3")
endif()




###############################################################################
# Include files
###############################################################################

if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)
find_package(ArrayFire)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("include")
include_directories("dependencies")
include_directories("dependencies/cutlass/include")
include_directories("dependencies/cutlass/tools/util/include")

add_subdirectory("src/tiny-cuda-nn")
add_subdirectory("src/nerf_volume")
add_subdirectory("dependencies/stbi")
add_subdirectory("dependencies/cutlass")
add_executable(test test.cu)
target_link_libraries(test PUBLIC ${CUDA_LIBRARIES} stbi tiny-cuda-nn nerf_volume)
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
# add_executable(nerf-volume main.cu)
# target_link_libraries(nerf-volume PUBLIC ${CUDA_LIBRARIES} stbi tiny-cuda-nn nerf_volume ArrayFire::af ArrayFire::afcuda ${OpenCV_LIBS})
# target_compile_options(nerf-volume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)