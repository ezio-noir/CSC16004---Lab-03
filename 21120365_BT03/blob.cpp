#include "algs.hpp"
#include "mat_ops.hpp"
#include "thread_pool.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// Initialize sigmas
void Blob::genSigmas() {
	sigmas.resize(nScales);
	sigmas[0] = initSigma;
	for (int i = 1; i < sigmas.size(); ++i)
		sigmas[i] = sigmas[i - 1] * k;
}

// Create scale space function for multi-threading
void Blob::createScaleSpaceTask(int s) {
	cout << "[+++ TASK +++] createScaleSpace, s = " << s << endl;

	// Scale by sigma (raw algorithm)
	if (scaleBySigma) {
		Mat filter = genNormLOGFilter(sigmas[s]);
		Mat laplResp = conv2d(mat, filter);
		scaleSpace[s] = applyEachElement(laplResp, [](float x) { return x * x; });
	}
	// Instead of scaling filter, we can subsample the image
	else {
		// We keep the filter unchange
		Mat filter = genNormLOGFilter(sigmas[0]);
		// Subsample the image
		Mat resized = Mat::zeros((int)((float)orgHeight / pow(k, s)), (int)((float)orgWidth / pow(k, s)), CV_32F);
		resize(mat, resized, resized.size(), 0, 0, INTER_AREA);
		// Calculate Laplacian response
		Mat laplResp = conv2d(resized, filter);
		// Resize to image original size
		Mat reresized = Mat::zeros(mat.size(), CV_32F);
		resize(applyEachElement(laplResp, [](float x) { return x * x; }), reresized, reresized.size(), 0, 0, INTER_CUBIC);
		scaleSpace[s] = reresized;

		/*Mat laplResp = conv2d(resize_bilinear(mat, (int)((float)orgHeight / pow(k, s)), (int)((float)orgWidth / pow(k, s))), filter);
		scaleSpace[s] = resize_nearest(applyEachElement(laplResp, [](float x) { return x * x; }), orgHeight, orgWidth);*/
	}

	cout << "[--- TASK ---] createScaleSpace, s = " << s << endl;
}

// Create scale space
void Blob::createScaleSpace() {
	scaleSpace.resize(nScales);
	// Initalize multi-thread pool
	ThreadPool tp(-1);
	for (int s = 0; s < nScales; ++s)
		tp.enqueue([=](int s) { createScaleSpaceTask(s); }, s);
}

// Suppress non-maxima within planes
void Blob::nonmaxSuppress2d() {
	// Iterate each scale
	for (int s = 0; s < nScales; ++s) {
		Mat suppressed = Mat::zeros(scaleSpace[s].rows, scaleSpace[s].cols, CV_32F);
		int m = suppressed.rows, n = suppressed.cols;
		// Iterate each pixel
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				bool isMaximum = true;
				for (int ii = max(i - 1, 0); isMaximum && ii <= min(i + 1, suppressed.rows - 1); ++ii) {
					for (int jj = max(j - 1, 0); isMaximum && jj <= min(j + 1, suppressed.cols - 1); ++jj) {
						if (ii == i && jj == j)
							continue;
						if (scaleSpace[s].at<float>(ii, jj) >= scaleSpace[s].at<float>(i, j))
							isMaximum = false;
					}
				}
				// A pixel keeps its value only it is a maximum
				if (isMaximum) {
					suppressed.at<float>(i, j) = scaleSpace[s].at<float>(i, j);
				}
			}
		}
		scaleSpace[s] = suppressed;
	}
}

// Find which 2D maxima have highest response along their own scale
void Blob::findCircles() {
	circles.resize(0);
	// The pixels at the lowest and highest scale are not considered
	for (int s = 1; s < nScales - 1; ++s) {
		int m = scaleSpace[s].rows, n = scaleSpace[s].cols;
		for (int i = border; i < m - border; ++i) {
			for (int j = border; j < n - border; ++j) {
				bool isMaximum = true;
				float centerVal = scaleSpace[s].at<float>(i, j);
				// Filter maximum by a threshold
				if (centerVal <= sqrRespThres)
					continue;
				// Check if it is 3x3x3 maximum
				for (int ss = s - 1; isMaximum && ss <= s + 1; ++ss) {
					for (int ii = i - 1; isMaximum && ii <= i + 1; ++ii) {
						for (int jj = j - 1; isMaximum && jj <= j + 1; ++jj) {
							if (ss == s && ii == i && jj == j)
								continue;
							if (scaleSpace[ss].at<float>(ii, jj) >= centerVal)
								isMaximum = false;
						}
					}
				}
				if (isMaximum) {
					// Here, the radius r = sigma[s] * sqrt(2) stays true even when we use
					// scale-image method, since down sizing an image by a factor k, then apply
					// a filter sigma is approximatly equivalent to filter the original image
					// with a filter k * sigma.
					circles.push_back(make_pair(make_pair(i, j), sigmas[s] * sqrt(2.0)));
				}
			}
		}
	}
}

// Eliminate the smaller circle in a pair, if the overlapping area surpass a threshold
void Blob::suppressOverlap(pair<Pt2d, float> &c1, pair<Pt2d, float> &c2) {
	float x1 = c1.first.first, y1 = c2.first.second, x2 = c2.first.first, y2 = c2.first.second;
	float r1 = c1.second, r2 = c2.second;
	float d = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) + (y2 - y1));
	
	// Case 1: 2 circles have no intersections
	if (d > r1 + r2)
		return;
	// Case 2: One circle completely contains the other
	if (d <= abs(r1 - r2)) {
		// Then we eliminate the smaller (inner) circle
		if (r1 > r2 && r2 / r1 > sqrt(ovlThres))
			c2.second = 0.0;
		else if (r1 / r2 > sqrt(ovlThres))
			c1.second = 0.0;
		return;
	}
	// Case 3: 2 circles have intersections
	float r = min(r1, r2), R = max(r1, r2);
	// Reference: https://mathworld.wolfram.com/Circle-CircleIntersection.html
	float overlapArea = r * r * acos((d * d + r * r - R * R) / (2 * d * r))
		+ R * R * acos((d * d + R * R - r * r) / (2 * d * R))
		- 0.5 * sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R));
	// If the overlapping area is greater than the threshold, remove the smaller circle
	if (overlapArea / (M_PI * r * r) > ovlThres) {
		if (r1 > r2)
			c2.second = 0.0;
		else
			c1.second = 0.0;
	}
}

// Eliminate circles that overlap bigger circles
void Blob::elimOverlaps() {
	vector<pair<Pt2d, float>> temp;
	for (int i = 0; i < circles.size(); ++i) {
		for (int j = i + 1; circles[i].second > 0.0 && j < circles.size(); ++j) {
			suppressOverlap(circles[i], circles[j]);
		}
		if (circles[i].second > 0.0)
			temp.push_back(circles[i]);
	}
	circles = temp;
}

// Draw a circle with given center and radius
void Blob::drawCircle(Pt2d center, float radius) {
	float step = 0.05;
	int i, j;
	Vec3b red = { 0, 0, 255 };
	// Travel counter-clockwise
	for (float theta = 0.0; theta <= 2 * M_PI; theta += step) {
		i = (int)(center.first + cos(theta) * radius);
		j = (int)(center.second + sin(theta) * radius);
		if (i >= 0 && i < mat.rows && j >= 0 && j < mat.cols)
			res.at<Vec3b>(i, j) = red;
	}
}

// Draw detected blobs
void Blob::drawCircles() {
	for (pair<Pt2d, float>& circle : circles) {
		drawCircle(circle.first, circle.second);
	}
}

// Do detect blobs
Mat Blob::doDetectBlob() {
	// Initialize sigmas
	genSigmas();
	// Create scale space
	createScaleSpace();
	// Suppress non-maxima within planes
	nonmaxSuppress2d();
	// Find which 2D maxima have highest response along their own scale
	findCircles();
	// If overlapping elimnation is enabled (disabled by default)
	if (ovlThres <= 1.0) {
		elimOverlaps();
	}
	// Draw detected blobs
	drawCircles();
	return res;
}

// Blob algorithm constructor
Blob::Blob(Mat orgImg, int nScales_, float initSigma_, float k_, float sqrRespThres_, int border_, float ovlThres_, bool scaleBySigma_) {
	this->orgImg = orgImg;
	this->mat = orgImg.clone();
	Mat imgAsInt;
	orgImg.convertTo(imgAsInt, CV_8U);
	cvtColor(imgAsInt, res, COLOR_GRAY2BGR);
	orgHeight = orgImg.rows;
	orgWidth = orgImg.cols;

	nScales = nScales_;
	initSigma = initSigma_;
	k = k_;
	sqrRespThres = sqrRespThres_;
	border = border_;
	ovlThres = ovlThres_;
	scaleBySigma = scaleBySigma_;
}

// Boilerplate code for initializing algorithm input and parameters
Mat Blob::detectBlob(Mat img, unordered_map<string, float> params) {
	// Initialize paramters
	int m = img.rows, n = img.cols;
	float initSigma = (params.find("init_sigma") != params.end()) ? params["init_sigma"] : 2.0;
	int nScales = (params.find("num_scales") != params.end()) ? (int)params["num_scales"] : 10;
	float k = (params.find("k") != params.end()) ? params["k"] : pow(2.0f, 0.25f);
	float sqrRespThres = (params.find("sqr_resp_thres") != params.end()) ? params["sqr_resp_thres"] : 1000.0;
	int border = (params.find("border") != params.end()) ? (int)params["border"] : 1;
	float ovlThres = (params.find("ovl_thres") != params.end()) ? params["ovl_thres"] : 1.01;
	bool scaleBySigma = (params.find("scale_sigma") != params.end() && (int)params["scale_sigma"]) ? true : false;

	// Convert input image to grayscale
	Mat src;
	if (img.channels() == 1) {
		img.convertTo(src, CV_32F);
	}
	else if (img.channels() == 3) {
		Mat grayImg;
		cvtColor(img, grayImg, COLOR_BGR2GRAY);
		grayImg.convertTo(src, CV_32F);
	}

	// Perform blur
	if (params.find("blur_ksize") != params.end() || params.find("blur_sigma") != params.end()) {
		int blur_ksize = (params.find("blur_ksize") != params.end()) ? (int)params["blur_ksize"] : 0;
		float blur_sigma = (params.find("blur_sigma") != params.end()) ? params["blur_sigma"] : 0.0;
		src = gaussianBlur(src, blur_ksize, blur_sigma);
	}

	// Detect blobs
	Blob blobDetector = Blob(src, nScales, initSigma, k, sqrRespThres, border, ovlThres, scaleBySigma);
	return blobDetector.doDetectBlob();
}