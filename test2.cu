#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <iostream>

int main()
{
    tcnn::pcg32 rng(42);
    tcnn::GPUMatrix<float> m1(3, 3);
    m1.initialize_xavier_uniform(rng);
    tcnn::GPUMatrix<float, tcnn::RM> m2 = m1.transposed();
    std::cout << m1.is_contiguous() << std::endl;
    std::cout << m2.is_contiguous() << std::endl;
    tcnn::MatrixView<float> m1View = m1.view();
    for (int i = 0; i < m1.rows(); ++i)
    {
        for (int j = 0; j < m1.cols(); ++j)
        {
            float val;
            cudaMemcpy(&val, &m1View(i, j), sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    tcnn::MatrixView<float> m2View = m2.view();
    for (int i = 0; i < m2.rows(); ++i)
    {
        for (int j = 0; j < m2.cols(); ++j)
        {
            float val;
            cudaMemcpy(&val, &m2View(i, j), sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}