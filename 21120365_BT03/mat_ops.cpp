#include "mat_ops.hpp"
#include <cmath>

#include <iostream>

using namespace cv;
using namespace std;

// 2D convolution
Mat conv2d(Mat mat, Mat kernel) {
	int m = mat.rows, n = mat.cols, i, j, ii, jj, ksize = kernel.rows / 2;
	Mat res = Mat::zeros(m, n, CV_32F);
	for (i = 0; i < m; ++i)
		for (j = 0; j < n; ++j)
			for (ii = max(i - ksize, 0); ii <= min(i + ksize, m - 1); ++ii)
				for (jj = max(j - ksize, 0); jj <= min(j + ksize, n - 1); ++jj)
					res.at<float>(i, j) += kernel.at<float>(ii - i + ksize, jj - j + ksize) * mat.at<float>(ii, jj);
	return res;
}

// Generates a Gaussian filter with given size and sigma
// Reference: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
Mat genGaussianFilter(int size, float sigma) {
	sigma = (sigma > 0) ? sigma : 1.0;
	// If size is not specified, then it is determined by sigma
	size = (size > 0) ? size : (int)(round)(2 * (1 + (sigma - 0.8) / 0.3) + 1) | 1;
	float s = sigma * sigma;
	Mat kernel = Mat::zeros(size, size, CV_32F);
	float sum = 0.0; // Stores sum of all elements for normalization
	int halfSize = size / 2;

	// Compute Gaussian at each pixel
	for (int x = -halfSize; x <= halfSize; ++x) {
		for (int y = -halfSize; y <= halfSize; ++y) {
			kernel.at<float>(x + halfSize, y + halfSize) = exp(-(float)(x * x + y * y) / (2 * s));
			sum += kernel.at<float>(x + halfSize, y + halfSize);
		}
	}

	// Normalize result
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			kernel.at<float>(i, j) /= sum;

	return kernel;
}

// Generate normalized LoG filter
Mat genNormLOGFilter(float sigma) {
	int halfSize = (int)ceil(sigma * 2.5f), size = 2 * halfSize + 1;
	Mat filter = Mat::zeros(size, size, CV_32F);
	float c = -1.0f / (M_PI * pow(sigma, 2.0f));
	for (int i = -halfSize; i <= halfSize; ++i) {
		for (int j = -halfSize; j <= halfSize; ++j) {
			float p = ((float)(i * i + j * j)) / (2 * sigma * sigma);
			filter.at<float>(halfSize + i, halfSize + j) = c * (1 - p) * exp(-p);
		}
	}
	return filter;
}

// Apply Gaussian filter on matrix
Mat gaussianBlur(Mat src, int ksize, float sigma) {
	return conv2d(src, genGaussianFilter(ksize, sigma));
}

// Element-wise operation
Mat applyElementwise(Mat x, Mat y, float (*op)(float, float)) {
	int m = x.rows, n = x.cols;
	Mat res = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			res.at<float>(i, j) = op(x.at<float>(i, j), y.at<float>(i, j));

	return res;
}

// Scalar value operation
Mat applyEachElement(Mat x, float (*op)(float)) {
	int m = x.rows, n = x.cols;
	Mat res = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			res.at<float>(i, j) = op(x.at<float>(i, j));
	return res;
}