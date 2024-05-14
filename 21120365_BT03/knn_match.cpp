#include "algs.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

// Match two images by kNN algorithm, with specified feature extration algorithm
double KnnMatch::match(Mat img1, Mat img2, int detector, Mat& res, unordered_map<string, float> params) {
	if (detector == DTT_HARRIS) {

	}
	else if (detector == DTT_BLOB) {

	}
	else if (detector == DTT_SIFT_DOG) {
		vector<KeyPoint> kp1, kp2;
		Mat desc1, desc2;
		SiftDOG::detectDOG(img1, params, kp1, desc1);
		SiftDOG::detectDOG(img2, params, kp2, desc2);

		BFMatcher matcher(NORM_L2);
		vector<vector<DMatch>> knnMatches;
		matcher.knnMatch(desc1, desc2, knnMatches, 2);

		vector<DMatch> matches;
		for (auto& knnMatch : knnMatches) {
			if (knnMatch.size() > 1 && knnMatch[0].distance < knnMatch[1].distance * KNN_MATCH_THRES) {
				matches.push_back(knnMatch[0]);
			}
		}

		// Draw matches
		drawMatches(img1, kp1, img2, kp2, matches, res, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		
		return (double)matches.size() / (double)knnMatches.size();
	}

	return -1.0;
}

