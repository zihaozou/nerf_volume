#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <nerf_volume/common_kernels.h>
#include <nerf_volume/texture.h>
#include <stbi/stbi_wrapper.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <json/json.hpp>
#include <fstream>
#include <memory>
int XDim = 256;
int YDim = 256;
int ZDim = 109;
using precision_t = tcnn::network_precision_t;

__global__ void toByte(uint32_t numEle, float *input, uint8_t *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEle)
    {
        output[idx] = (uint8_t)(fmaxf(fminf(input[idx], 1.0f), 0.0f) * 255.0f);
    }
}

bool loadRAW(float **dataBuffer, const char *fname, size_t size)
{
    // Load RAW file into memory
    uint8_t *dataBufferChar = nullptr;
    FILE *fp = fopen(fname, "rb");
    if (fp == 0)
    {
        std::cout << "open file error" << std::endl;
        return false;
    }
    dataBufferChar = (uint8_t *)malloc(size);
    *dataBuffer = (float *)malloc(size * sizeof(float));
    fread(dataBufferChar, 1, size, fp);
    fclose(fp);
    float *vdest = *dataBuffer;
    uint8_t *vsrc = dataBufferChar;
    for (int n = 0; n < XDim * YDim * ZDim; n++)
        vdest[n] = float(vsrc[n]) / 255.0f;
    free(dataBufferChar);
    std::cout << "read file success" << std::endl;
    return true;
}

int main(int argc, char *argv[])
{
    nlohmann::json config;
    std::ifstream f{"/home/zihao/Desktop/workspace/nerf_volume/data/config_hash.json"};
    config = nlohmann::json::parse(f, nullptr, true, /*skip_comments=*/true);
    float *dataBuffer = nullptr;
    loadRAW(&dataBuffer, "/home/zihao/Desktop/workspace/nerf_volume/data/head256x256x109", XDim * YDim * ZDim);
    cudaTextureObject_t tex_obj = nerf_volume::create3DTex(dataBuffer, XDim, YDim, ZDim);
    // training
    uint32_t n_coords = XDim * YDim;
    uint32_t n_coords_padded = tcnn::next_multiple(n_coords, tcnn::batch_size_granularity);

    tcnn::GPUMemory<float> sampled_image(n_coords);
    tcnn::GPUMemory<float> xyz(n_coords_padded * 3);

    std::vector<float> xyz_host(n_coords * 3);
    for (int y = 0; y < YDim; y++)
    {
        for (int x = 0; x < XDim; x++)
        {
            int idx = (y * XDim + x) * 3;
            xyz_host[idx + 0] = (float)(x) / (float)(XDim - 1);
            xyz_host[idx + 1] = (float)(y) / (float)(YDim - 1);
            xyz_host[idx + 2] = 0.55f;
        }
    }

    xyz.copy_from_host(xyz_host.data());

    const uint32_t batch_size = 1 << 16;
    const uint32_t n_training_steps = 100000;
    const uint32_t n_input_dims = 3;  // 3-D MRI
    const uint32_t n_output_dims = 1; // single value
    tcnn::GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
    tcnn::GPUMatrix<float> inference_batch(xyz.data(), n_input_dims, n_coords_padded);
    tcnn::GPUMemory<uint8_t> inferImageGPU(n_coords);
    std::vector<uint8_t> inferImage(n_coords);
    cudaStream_t inference_stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    cudaStream_t training_stream = inference_stream;
    tcnn::pcg32 rng{42};
    tcnn::GPUMatrix<float> training_target(n_output_dims, batch_size);
    tcnn::GPUMatrix<float> training_batch(n_input_dims, batch_size);
    tcnn::GPUMatrix<float> training_batch_int(n_input_dims, batch_size);
    nlohmann::json encoding_opts = config.value("encoding", nlohmann::json::object());
    nlohmann::json loss_opts = config.value("loss", nlohmann::json::object());
    nlohmann::json optimizer_opts = config.value("optimizer", nlohmann::json::object());
    nlohmann::json network_opts = config.value("network", nlohmann::json::object());
    std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(loss_opts)};
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizer_opts)};
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);
    auto trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

    tcnn::linear_kernel(nerf_volume::convertToInt, 0, inference_stream, n_coords, xyz.data(), training_batch_int.data(), XDim, YDim, ZDim);
    tcnn::linear_kernel(nerf_volume::matchTarget, 0, inference_stream, n_coords, training_target.data(), tex_obj, training_batch_int.data());
    cudaStreamSynchronize(inference_stream);
    auto ref_filename = "reference.jpg";
    tcnn::linear_kernel(toByte, 0, inference_stream, n_coords, training_target.data(), inferImageGPU.data());
    cudaStreamSynchronize(inference_stream);
    inferImageGPU.copy_to_host(inferImage.data());
    save_stbi(inferImage.data(), XDim, YDim, 1, ref_filename);
    for (uint32_t i = 0; i < n_training_steps; i++)
    {
        tcnn::generate_random_uniform<float>(training_stream, rng, batch_size * n_input_dims, training_batch.data());
        tcnn::linear_kernel(nerf_volume::convertToInt, 0, training_stream, batch_size, training_batch.data(), training_batch_int.data(), XDim, YDim, ZDim);
        tcnn::linear_kernel(nerf_volume::matchTarget, 0, training_stream, batch_size, training_target.data(), tex_obj, training_batch_int.data());
        auto contex = trainer->training_step(training_stream, training_batch, training_target);
        if (i % 1000 == 0)
        {
            cudaStreamSynchronize(inference_stream);
            std::cout << "step " << i << ": " << trainer->loss(training_stream, *contex) << std::endl;
            network->inference(inference_stream, inference_batch, prediction);
            auto filename = "step:" + std::to_string(i) + "_inference.jpg";
            tcnn::linear_kernel(toByte, 0, inference_stream, n_coords, prediction.data(), inferImageGPU.data());
            cudaStreamSynchronize(inference_stream);
            inferImageGPU.copy_to_host(inferImage.data());
            save_stbi(inferImage.data(), XDim, YDim, 1, filename.c_str());
        }
    }
}