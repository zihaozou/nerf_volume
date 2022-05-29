#include <nerf_volume/common_kernels.h>
NV_NAME_SPACE_BEGIN
__global__ void extractLayerKernel(uint8_t *data, uint32_t XDim, uint32_t YDim, cudaTextureObject_t texture, float z)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= XDim * YDim)
        return;
    idx += 1;
    float x = ((float)(idx % XDim) / (float)XDim);
    float y = ((float)(idx / XDim) / (float)YDim);
    float texel = tex3D<float>(texture, x, y, z);

    data[idx] = (uint8_t)(fmaxf(fminf(texel, 1.0f), 0.0f) * 255.0f);
}
NV_NAME_SPACE_END