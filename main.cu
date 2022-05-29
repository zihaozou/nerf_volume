#include <tiny-cuda-nn/common_device.h>
#include <nerf_volume/common_kernels.h>
#include <nerf_volume/texture.h>
#include <stbi/stbi_wrapper.h>
#include <iostream>
#include <vector>
#include <stdio.h>

float *dataBuffer = nullptr;
int XDim = 256;
int YDim = 256;
int ZDim = 109;

bool loadRAW(const char *fname, size_t size)
{
    // Load RAW file into memory
    char *dataBufferChar = nullptr;
    FILE *fp = fopen(fname, "rb");
    if (fp == 0)
    {
        std::cout << "open file error" << std::endl;
        return false;
    }
    dataBufferChar = (char *)malloc(size);
    dataBuffer = (float *)malloc(size * sizeof(float));
    fread(dataBufferChar, 1, size, fp);
    fclose(fp);
    float *vdest = dataBuffer;
    char *vsrc = dataBufferChar;
    for (int n = 0; n < XDim * YDim * ZDim; n++)
        *vdest++ = float(*vsrc++) / 255.0f;
    free(dataBufferChar);
    std::cout << "read file success" << std::endl;
    return true;
}
int main(int argc, char *argv[])
{
    loadRAW("/home/zihao/Desktop/workspace/nerf_volume/data/head256x256x109", XDim * YDim * ZDim);
    cudaTextureObject_t tex_obj = nerf_volume::create3DTex(dataBuffer, XDim, YDim, ZDim);
    free(dataBuffer);
    uint8_t *layerData;
    CUDA_CHECK_THROW(cudaMalloc(&layerData, XDim * YDim * sizeof(uint8_t)));
    nerf_volume::extractLayerKernel<<<256, 256>>>(layerData, XDim, YDim, tex_obj, 0.5f);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    uint8_t *layerDataHost;
    CUDA_CHECK_THROW(cudaMallocHost(&layerDataHost, XDim * YDim * sizeof(uint8_t)));
    CUDA_CHECK_THROW(cudaMemcpy(layerDataHost, layerData, XDim * YDim * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    save_stbi(layerDataHost, XDim, YDim, 1, "test.jpg");
    CUDA_CHECK_THROW(cudaFree(layerData));
    CUDA_CHECK_THROW(cudaFreeHost(layerDataHost));
}