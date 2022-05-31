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
    float val = tex3D<float>(texture, xyz[xyzIdx], xyz[xyzIdx + 1], xyz[xyzIdx + 2]);
    data[i] = val;
};
NV_NAME_SPACE_END