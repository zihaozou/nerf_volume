#pragma once

#include <iostream>
#include <nerf_volume/common.h>
NV_NAME_SPACE_BEGIN
__global__ void extractLayerKernel(uint8_t *data, uint32_t XDim, uint32_t YDim, cudaTextureObject_t texture, float z);
NV_NAME_SPACE_END