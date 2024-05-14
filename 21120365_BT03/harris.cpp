#include "algs.hpp"
#include "mat_ops.hpp"
#include "fast_conv.hpp"

using namespace std;
using namespace cv;

// Mark detected corners
void Harris::drawPoints() {
	Vec3b red = { 0, 0, 255 };
	for (Point& point : kps) {
		res.at<Vec3b>(point.y, point.x) = red;
	}
}

// Harris algorithm
void Harris::doDetectHarris() {
	// Calculate derivatives
	Mat sobel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat sobel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat dx, dy;
	{
		FastConv fc(orgImg, 200, sobel_x);
		dx = fc.fastConv2d();
	}
	{
		FastConv fc(orgImg, 200, sobel_y);
		dy = fc.fastConv2d();
	}

	// Calculate Ixx, Ixy and Iyy
	Mat Ixx = applyElementwise(dx, dx, [](float x, float y) { return x * y; });
	Mat Ixy = applyElementwise(dx, dy, [](float x, float y) { return x * y; });
	Mat Iyy = applyElementwise(dy, dy, [](float x, float y) { return x * y; });

	int halfSize = window.rows / 2;
	kps.resize(0);
	// Iterate through each pixel
	for (int i = 0; i < orgHeight; ++i) {
		for (int j = 0; j < orgWidth; ++j) {
			// Caculate M matrix
			float Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
			for (int ii = max(0, i - halfSize); ii <= min(orgHeight - 1, i + halfSize); ++ii) {
				for (int jj = max(0, j - halfSize); jj <= min(orgWidth - 1, j + halfSize); ++jj) {
					Sxx += Ixx.at<float>(ii, jj) * window.at<float>(ii - i + halfSize, jj - j + halfSize);
					Sxy += Ixy.at<float>(ii, jj) * window.at<float>(ii - i + halfSize, jj - j + halfSize);
					Syy += Iyy.at<float>(ii, jj) * window.at<float>(ii - i + halfSize, jj - j + halfSize);
				}
			}
			// Calculate response
			float resp = (Sxx * Syy - Sxy * Sxy) - alpha * (Sxx + Syy) * (Sxx + Syy);
			// Filter response
			if (resp > thres) {
				kps.push_back(Point(j, i));
			}
		}
	}

	// Draw key points
	drawPoints();
}

// Harris algorithm constructor
Harris::Harris(Mat orgImg_, float alpha_, float thres_, int windowSize_) {
	this->orgImg = orgImg_;
	orgHeight = orgImg.rows;
	orgWidth = orgImg.cols;
	Mat imgAsInt;
	orgImg.convertTo(imgAsInt, CV_8U);
	cvtColor(imgAsInt, res, COLOR_GRAY2BGR);
	alpha = alpha_;
	thres = thres_;
	windowSize = windowSize_;
	window = genGaussianFilter(windowSize, 1.0);
}

// Boilerplate code for initializing algorithm input and parameters
Mat Harris::detectHarris(Mat img, unordered_map<string, float> params) {
	// Initialize paramters
	float alpha_ = (params.find("alpha") != params.end()) ? params["alpha"] : 0.04;
	float thres_ = (params.find("thres") != params.end()) ? params["thres"] : 10000.0;
	int windowSize_ = (params.find("window_size") != params.end()) ? (int)params["window_size"] : 5;

	// Convert input image to grayscale
	Mat src;
	if (img.channels() == 1) {
		img.convertTo(src, CV_32F);
	}
	else if (img.channels() == 3) {
		Mat imgGray;
		cvtColor(img, imgGray, COLOR_BGR2GRAY);
		imgGray.convertTo(src, CV_32F);
	}

	// Perform blur
	src = gaussianBlur(src, 3, 1.0);
	Harris harris(src, alpha_, thres_, windowSize_);

	// Detect corners
	harris.doDetectHarris();
	return harris.res;
}