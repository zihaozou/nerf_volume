#include <nerf_volume/common_kernels.h>
NV_NAME_SPACE_BEGIN
__global__ void extractLayerKernel(uint8_t *data, uint32_t XDim, uint32_t YDim, cudaTextureObject_t texture, float z)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= XDim * YDim)
        return;
    float x = ((float)(idx % XDim) / (float)(XDim - 1));
    float y = ((float)(idx / XDim) / (float)(YDim - 1));
    float texel = tex3D<float>(texture, x, y, z);

    data[idx] = (uint8_t)(fmaxf(fminf(texel, 1.0f), 0.0f) * 255.0f);
};

__global__ void matchTarget(uint32_t numData, float *__restrict__ data, cudaTextureObject_t texture, float *__restrict__ xyz)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numData)
        return;
    uint32_t xyzIdx = i * 3;
    float val = tex3D<float>(texture, xyz[xyzIdx] + 0.5, xyz[xyzIdx + 1] + 0.5, xyz[xyzIdx + 2] + 0.5);
    data[i] = val;
};

__global__ void convertToInt(uint32_t numData, float *__restrict__ src, float *__restrict__ dst, uint32_t XDim, uint32_t YDim, uint32_t ZDim)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numData)
        return;
    uint32_t xyzIdx = i * 3;
    dst[xyzIdx] = (float)(int)(src[xyzIdx] * (float)(XDim - 1));
    dst[xyzIdx + 1] = (float)(int)(src[xyzIdx + 1] * (float)(YDim - 1));
    dst[xyzIdx + 2] = (float)((int)(src[xyzIdx + 2] * (float)(ZDim - 1)));
    src[xyzIdx] = (dst[xyzIdx]) / (float)(XDim - 1);
    src[xyzIdx + 1] = (dst[xyzIdx + 1]) / (float)(YDim - 1);
    src[xyzIdx + 2] = (dst[xyzIdx + 2]) / (float)(ZDim - 1);
}

NV_NAME_SPACE_END