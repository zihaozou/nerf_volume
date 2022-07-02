add_library(nerf_volume STATIC common_kernels.cu texture.cu)
target_link_libraries(nerf_volume PUBLIC ${CUDA_LIBRARIES} cuda cublas tiny-cuda-nn)
target_compile_options(nerf_volume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)