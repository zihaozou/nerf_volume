#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <nerf_volume/common.h>
#include <json/json.hpp>
#include <memory>
#include <functional>
#include <vector>
NV_NAME_SPACE_BEGIN

template <typename T>
class DEQNetwork : public tcnn::Network<float, T>
{
public:
    DEQNetwork(std::shared_ptr<tcnn::Encoding<T>> encoding, uint32_t n_output_dims, const nlohmann::json &network) : m_encoding{encoding}
    {
        encoding->set_alignment(tcnn::minimum_alignment(network));

        nlohmann::json local_network_config = network;
        local_network_config["n_input_dims"] = m_encoding->padded_output_width();
        local_network_config["n_output_dims"] = n_output_dims;
        m_network.reset(tcnn::create_network<T>(local_network_config));
    };

    DEQNetwork(uint32_t n_dims_to_encode, uint32_t n_output_dims, const nlohmann::json &encoding, const nlohmann::json &network) : DEQNetwork{std::shared_ptr<tcnn::Encoding<T>>{tcnn::create_encoding<T>(n_dims_to_encode, encoding)}, n_output_dims, network} {};

    virtual ~DEQNetwork(){};
    void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float> &input, tcnn::GPUMatrixDynamic<T> &output, bool use_inference_params = true) override
    {
        tcnn::GPUMatrixDynamic<T> network_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};

        m_encoding->inference_mixed_precision(stream, input, network_input, use_inference_params);
        m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
    }

private:
    std::unique_ptr<tcnn::Network<T>> m_network;
    std::shared_ptr<tcnn::Encoding<T>> m_encoding;
    struct ForwardContext : public tcnn::Context
    {
        tcnn::GPUMatrixDynamic<T> network_input;
        std::unique_ptr<tcnn::Context> encoding_ctx;
        std::unique_ptr<tcnn::Context> network_ctx;
    };
    std::unique_ptr<tcnn::GPUMatrix<T>> addreson_forward(cudaStream_t stream, std::function<tcnn::GPUMatrixDynamic<T> *(const tcnn::GPUMatrixDynamic<T> &)> func, const tcnn::GPUMatrixDynamic<T> &x0, uint32_t m = 5, float lam = 1e-4, uint32_t max_iter = 50, float tol = 1e-2, float beta = 1.0)
    {
        std::vector<tcnn::GPUMatrix<T>> X(m);
        for (uint32_t i = 0; i < m; i++)
        {
            X[i] = tcnn::GPUMatrix<T>{x0.m(), x0.n(), stream};
        }
        std::vector<tcnn::GPUMatrix<T>> F(m);
        for (uint32_t i = 0; i < m; i++)
        {
            F[i] = tcnn::GPUMatrix<T>{x0.m(), x0.n(), stream};
        }
    }
};
NV_NAME_SPACE_END