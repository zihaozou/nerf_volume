#include <iostream>
#include <nerf_volume/nerf_volume.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <arrayfire.h>
#include <af/cuda.h>
int main()
{
    af::setBackend(AF_BACKEND_CUDA);
    std::vector<cv::Mat> imageLst;
    cv::imreadmulti(std::filesystem::path("../data/2101_Hep3B_009-1-1.tiff").c_str(), imageLst);
    af::array imageArray = af::array(imageLst[0].rows, imageLst[0].cols, imageLst.size(), u8);
    for (int i = 0; i < imageLst.size(); i++)
    {
        cv::Mat tempMat;
        cv::transpose(imageLst[i], tempMat);
        imageArray(af::span, af::span, i) = af::array(imageLst[0].rows, imageLst[0].cols, tempMat.data);
    }
    af::eval(imageArray);
    af::sync();
    af::array floatImageArray = imageArray.as(f32) / 255.0;
    af::eval(floatImageArray);
    af::sync();
}