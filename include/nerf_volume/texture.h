#pragma once
#include <iostream>
#include <tiny-cuda-nn/common_device.h>
#include <nerf_volume/common.h>
NV_NAME_SPACE_BEGIN
template <typename T>
cudaTextureObject_t create3DTex(T *dataCPU, uint32_t XDim, uint32_t YDim, uint32_t ZDim)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaArray_t array;
    CUDA_CHECK_THROW(cudaMalloc3DArray(&array, &channelDesc, make_cudaExtent(XDim, YDim, ZDim), 0));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(dataCPU, XDim * sizeof(T), XDim, YDim);
    copyParams.dstArray = array;
    copyParams.extent = make_cudaExtent(XDim, YDim, ZDim);
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK_THROW(cudaMemcpy3D(&copyParams));
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    cudaTextureObject_t tex;
    CUDA_CHECK_THROW(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    return tex;
}

void destory3DTex(cudaTextureObject_t tex)
{
    cudaResourceDesc resDesc;
    cudaGetTextureObjectResourceDesc(&resDesc, tex);
    cudaArray_t array = resDesc.res.array.array;
    CUDA_CHECK_THROW(cudaFreeArray(array));
    CUDA_CHECK_THROW(cudaDestroyTextureObject(tex));
}
NV_NAME_SPACE_END