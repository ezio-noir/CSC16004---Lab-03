#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef std::pair<int, int> Pt2d;

cv::Mat conv2d(cv::Mat mat, cv::Mat kernel);
cv::Mat genGaussianFilter(int size, float sigma);
cv::Mat genNormLOGFilter(float sigma);

cv::Mat gaussianBlur(cv::Mat src, int ksize, float sigma);

cv::Mat applyElementwise(cv::Mat x, cv::Mat y, float (*op)(float, float));
cv::Mat applyEachElement(cv::Mat x, float (*op)(float));
