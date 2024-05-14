#pragma once

#include "thread_pool.hpp"
#include "mat_ops.hpp"

using namespace std;
using namespace cv;

// Fast convolution by dividing matrix to smaller regions
class FastConv {
private:
	Mat src;
	int splitSize;
	Mat filter;

private:
	int m, n;
	int ksize;
	int halfSize;
	vector<pair<pair<int, int>, pair<int, int>>> regions;
	vector<float> regSum;

public:
	Mat dst;

private:
	void fastConv2dTask(int reg) {
		int i_tl = regions[reg].first.first, j_tl = regions[reg].first.second, i_br = regions[reg].second.first, j_br = regions[reg].second.second;
		for (int i = i_tl; i < i_br; ++i) {
			for (int j = j_tl; j < j_br; ++j) {
				for (int ii = max(0, i - halfSize); ii <= min(m - 1, i + halfSize); ++ii) {
					for (int jj = max(0, j - halfSize); jj <= min(n - 1, j + halfSize); ++jj) {
						dst.at<float>(i, j) += src.at<float>(ii, jj) * filter.at<float>(ii - i + halfSize, jj - j + halfSize);
					}
				}
			}
		}
	}

public:
	Mat fastConv2d() {
		regions.resize(0);
		for (int i = 0; i < m; i = i + splitSize) {
			for (int j = 0; j < n; j = j + splitSize) {
				int iStart = i, jStart = j, iEnd = min(m, i + splitSize), jEnd = min(n, j + splitSize);
				regions.push_back(make_pair(make_pair(iStart, jStart), make_pair(iEnd, jEnd)));
			}
		}
		regSum.resize(regions.size(), 0.0);
		{
			ThreadPool tp(-1);
			for (int r = 0; r < regions.size(); ++r) {
				tp.enqueue([=](int reg) { fastConv2dTask(reg); }, r);
			}
		}
		return dst;
	}

public:
	FastConv(Mat src_, int splitSize_, Mat filter_) {
		this->src = src_;
		this->splitSize = splitSize_;
		this->dst = Mat::zeros(src.rows, src.cols, CV_32F);
		this->filter = filter_;
		m = src.rows, n = src.cols;
		ksize = filter.rows;
		halfSize = ksize / 2;
	}
};