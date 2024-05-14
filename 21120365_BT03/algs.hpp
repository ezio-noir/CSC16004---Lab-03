#pragma once

#include "mat_ops.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <unordered_map>
#include <string>

#define DTT_HARRIS 0
#define DTT_BLOB 1
#define DTT_SIFT_DOG 2

using Octave = std::vector<cv::Mat>;

const float SIFT_INIT_SIGMA = 0.5;
const int SIFT_NUM_HIST_BINS = 36;
const float SIFT_ORI_RADIUS = 4.5;
const float SIFT_ORI_SIG_FCTR = 1.5;
const float SIFT_ORI_PEAK_RATIO = 0.8;
const int SIFT_DESCR_WIDTH = 4;
const int SIFT_DESCR_HIST_BINS = 8;
const int SIFT_DESCR_SIZE = 4 * 4 * 8;
const float SIFT_DESCR_SCL_FTR = 3.0;
const float SIFT_DESCR_MAG_THR = 0.2;
const float SIFT_INT_DESCR_FCTR = 512.0;
const float KNN_MATCH_THRES = 0.7f;

class Harris {
private:
	cv::Mat orgImg;
	int orgHeight;
	int orgWidth;
	float alpha;
	float thres;
	int windowSize;

private:
	cv::Mat window;
	std::vector<cv::Point> kps;

public:
	cv::Mat res;

private:
	void drawPoints();

public:
	void doDetectHarris();

public:
	Harris(cv::Mat orgImg_, float alpha_, float thres_, int windowSize_);

public:
	static cv::Mat detectHarris(cv::Mat img, std::unordered_map<std::string, float> params);
};

class Blob {
private:
	cv::Mat orgImg;
	int orgHeight;
	int orgWidth;
	int nScales;
	float initSigma;
	float k;
	float sqrRespThres;
	int border;
	float ovlThres;
	bool scaleBySigma;

private:
	cv::Mat mat;
	std::vector<float> sigmas;
	std::vector<cv::Mat> scaleSpace;
	std::vector<std::pair<Pt2d, float>> circles;

public:
	cv::Mat res;

private:
	void genSigmas();
	void createScaleSpaceTask(int s);
	void createScaleSpace();
	void nonmaxSuppress2d();
	void findCircles();
	void suppressOverlap(std::pair<Pt2d, float> &c1, std::pair<Pt2d, float> &c2);
	void elimOverlaps();
	void drawCircle(Pt2d center, float radius);
	void drawCircles();

public:
	cv::Mat doDetectBlob();

public:
	Blob(cv::Mat normImg, int nScales_, float initSigma, float k_, float sqrRespThres_, int border_, float ovlThres_, bool scaleBySigma_);

public:
	static cv::Mat detectBlob(cv::Mat img, std::unordered_map<std::string, float> params);
};

class SiftDOG {
private:
	float prepSigma;

private:
	cv::Mat orgImg;
	int orgHeight;
	int orgWidth;
	int nOctaves;
	int nLayers;
	float initSigma;
	float step;
	float contrastThres;
	float edgeThres;
	int nIters;
	int border;

private:
	cv::Mat mat;
	cv::Mat floatMat;
	std::vector<float> sigmas;
	std::vector<Octave> gPyr;
	std::vector<Octave> dPyr;
	std::vector<std::vector<cv::KeyPoint>> keyPts;
	std::vector<cv::KeyPoint> allKeyPts;
	cv::Mat descriptors;
	cv::Mat res;

private:
	void init();
	void genSigmas();
	void createGaussianPyramid();
	void createDOGPyramidTask(int otv, int layer);
	void createDOGPyramid();
	bool elimEdgeResp(int kpi, int kpj, int kps, int otv);
	bool localizeExtremum(int& kpi, int& kpj, int& kps, int otv, cv::KeyPoint& kp);
	float calcOrientation(int kpi, int kpj, int kps, int otv, int radius, float sigma, std::vector<float>& hist);
	void findLocalExtremaTask(int otv);
	void findLocalExtrema();
	void sortAndRemoveDuplicateKeypoints();
	void denormalize();
	void calcDescriptor(cv::Mat& curr, cv::Point2f ptd, float ori, float scale, int kpIdx);
	void calcDescriptors();

private:
	void doDetectDOG();

public:
	cv::Mat getResMat();
	std::vector<cv::KeyPoint> getKeyPoints();
	cv::Mat getDescriptors();

public:
	SiftDOG(cv::Mat img, int nOctaves_, int nLayers_, float initSigma_, float constrastThres_, float edgeThres_, int nIters_, int border_, float prepSigma_);

public:
	static cv::Mat detectDOG(cv::Mat img, std::unordered_map<std::string, float> params, std::vector<cv::KeyPoint>& kps, cv::Mat& descriptors);
};

class KnnMatch {
public:
	static double match(cv::Mat img1, cv::Mat img2, int detector, cv::Mat& res, std::unordered_map<std::string, float> params);
};