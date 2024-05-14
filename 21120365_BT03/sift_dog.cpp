//// Note:
//// https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5
//// https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp#L122
//// https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp#L501
//// https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp#L224
//// https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1


#include "algs.hpp"
#include "thread_pool.hpp"
#include "fast_conv.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


// Display a matrix as image (for debugging only)
void convertAndDisplay(Mat& img) {
	namedWindow("res", WINDOW_AUTOSIZE);
	Mat disp;
	img.convertTo(disp, CV_8U);
	imshow("res", disp);
	waitKey(0);
}

// 2x upsampling the orignal image
void SiftDOG::init() {
	float sigmaDiff = sqrt(max(0.01, initSigma * initSigma - prepSigma * prepSigma * 4.0));
	floatMat = Mat::zeros(orgHeight * 2, orgWidth * 2, CV_32F);
	resize(mat, floatMat, floatMat.size(), 0.0, 0.0, INTER_LINEAR);
	floatMat = gaussianBlur(floatMat, -1, sigmaDiff);
}

// Initialize sigmas
void SiftDOG::genSigmas() {
	sigmas.resize(nLayers + 3);
	sigmas[0] = initSigma;
	float prevSig;
	for (int i = 1; i < sigmas.size(); ++i)
		sigmas[i] = initSigma * sqrt(pow(step, 2.0 * (float)i) - pow(step, 2.0 * (float)(i - 1)));
}

// Create Gaussian pyramid
void SiftDOG::createGaussianPyramid() {
	gPyr.resize(nOctaves, Octave(nLayers + 3));
	gPyr[0][0] = floatMat.clone();
	for (int o = 0; o < nOctaves; ++o) {
		for (int l = 0; l < gPyr[o].size(); ++l) {
			if (o == 0 && l == 0)
				continue;
			Mat& current = gPyr[o][l];
			if (l == 0) {
				Mat& sample = gPyr[o - 1][nLayers];
				resize(sample, current, Size(sample.cols / 2, sample.rows / 2), 0.0, 0.0, INTER_NEAREST);
				continue;
			}
			{
				// Fast convolution by dividing image into smaller crops
				FastConv fc(gPyr[o][l - 1], 50, genGaussianFilter(-1, sigmas[l]));
				current = fc.fastConv2d();
			}
			//current = gaussianBlur(gPyr[o][l - 1], -1, sigmas[l]);
			//GaussianBlur(gPyr[o][l - 1], current, Size(), sigmas[l]);
		}
	}
}

// Create DOG pyramid function for multi-threading
void SiftDOG::createDOGPyramidTask(int otv, int layer) {
	cout << "[+++ TASK +++] createDOGPyramidTask, otv = " << otv << ", layer = " << layer << endl;
	dPyr[otv][layer] = applyElementwise(gPyr[otv][layer + 1], gPyr[otv][layer], [](float x, float y) { return x - y; });
	cout << "[--- TASK ---] createDOGPyramidTask, otv = " << otv << ", layer = " << layer << endl;
}

// Create DOG pyramid
void SiftDOG::createDOGPyramid() {
	dPyr.resize(nOctaves, Octave(nLayers + 2));
	ThreadPool tp(-1);
	for (int o = 0; o < nOctaves; ++o) {
		for (int l = 0; l < dPyr[o].size(); ++l) {
			// Each subtraction can be done idependently
			tp.enqueue([=](int otv, int layer) { createDOGPyramidTask(otv, layer); }, o, l);
		}
	}
}

// Eliminate edge response
bool SiftDOG::elimEdgeResp(int kpi, int kpj, int kps, int otv) {
	Mat& lower = dPyr[otv][kps - 1];
	Mat& curr = dPyr[otv][kps];
	Mat& upper = dPyr[otv][kps + 1];
	float secondDerivScale = 1.0 / 255.0;
	float crossDerivScale = 0.25 / 255.0;
	float dxx = curr.at<float>(kpi, kpj + 1) - 2.0 * curr.at<float>(kpi, kpj) + curr.at<float>(kpi, kpj - 1) * secondDerivScale;
	float dyy = curr.at<float>(kpi + 1, kpj) - 2.0 * curr.at<float>(kpi, kpj) + curr.at<float>(kpi - 1, kpj) * secondDerivScale;
	float dxy = (curr.at<float>(kpi + 1, kpj + 1) - curr.at<float>(kpi + 1, kpj - 1) - curr.at<float>(kpi - 1, kpj + 1) + curr.at<float>(kpi - 1, kpj - 1)) * crossDerivScale;
	float traceH = dxx + dyy, detH = dxx * dyy - dxy * dxy;
	if (detH <= 0)
		return false;
	return (traceH * traceH * edgeThres < (edgeThres + 1) * (edgeThres + 1) * detH);
}

// Localizing a extremum coords into sub-pixel accuracy
bool SiftDOG::localizeExtremum(int& kpi, int& kpj, int& kps, int otv, KeyPoint& kp) {
	float di, dj, ds;
	int it = 0;
	float firstDerivScale = 0.5 / 255.0;
	float secondDerivScale = 1.0 / 255.0;
	float crossDerivScale = 0.25 / 255.0;

	for (; it < nIters; ++it) {
		Mat& lower = dPyr[otv][kps - 1];
		Mat& curr = dPyr[otv][kps];
		Mat& upper = dPyr[otv][kps + 1];

		// First order derivatives
		Vec3d grad(
			(curr.at<float>(kpi, kpj + 1) - curr.at<float>(kpi, kpj - 1)) * firstDerivScale,
			(curr.at<float>(kpi + 1, kpj) - curr.at<float>(kpi - 1, kpj)) * firstDerivScale,
			(upper.at<float>(kpi, kpj) - lower.at<float>(kpi, kpj)) * firstDerivScale
			);

		// Second order derivatives
		float dxx = curr.at<float>(kpi, kpj + 1) - 2.0 * curr.at<float>(kpi, kpj) + curr.at<float>(kpi, kpj - 1) * secondDerivScale;
		float dyy = curr.at<float>(kpi + 1, kpj) - 2.0 * curr.at<float>(kpi, kpj) + curr.at<float>(kpi - 1, kpj) * secondDerivScale;
		float dss = upper.at<float>(kpi, kpj) - 2.0 * curr.at<float>(kpi, kpj) + lower.at<float>(kpi, kpj) * secondDerivScale;
		float dxy = (curr.at<float>(kpi + 1, kpj + 1) - curr.at<float>(kpi + 1, kpj - 1) - curr.at<float>(kpi - 1, kpj + 1) + curr.at<float>(kpi - 1, kpj - 1)) * crossDerivScale;
		float dxs = (upper.at<float>(kpi, kpj + 1) - upper.at<float>(kpi, kpj - 1) - lower.at<float>(kpi, kpj + 1) + lower.at<float>(kpi, kpj - 1)) * crossDerivScale;
		float dys = (upper.at<float>(kpi + 1, kpj) - upper.at<float>(kpi - 1, kpj) - lower.at<float>(kpi + 1, kpj) + lower.at<float>(kpi - 1, kpj)) * crossDerivScale;

		// Solve least-square for minimizing error
		Matx33d hess(
			dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);
		Vec3d delta = hess.solve(grad, DECOMP_LU);
		di = delta[1];
		dj = delta[0];
		ds = delta[2];

		// If the delta is too small, then the pixel coords shall not be modified anymore
		if (abs(di) < 0.5 && abs(dj) < 0.5 && abs(ds) < 0.5)
			break;
		// Overflowing/out of border
		if (abs(di) > 1e9 || abs(dj) > 1e9 || abs(ds) > 1e9)
			return false;
		// Update pixel coords
		kpi = (int)round(kpi - di);
		kpj = (int)round(kpj - dj);
		kps = (int)round(kps - ds);
		// If the coords is outside of scale space
		if (kps < 1 || kps > nLayers || kpi < border || kpi >= curr.rows - border || kpj < border || kpj >= curr.cols - border)
			return false;
	}
	// After the number of iterations, the pixel coords still not converge
	if (it >= nIters)
		return false;

	Mat& lower = dPyr[otv][kps - 1];
	Mat& curr = dPyr[otv][kps];
	Mat& upper = dPyr[otv][kps + 1];
	float ct = curr.at<float>(kpi, kpj) * 1.0 / 255.0 - 0.5 * (
		di * (curr.at<float>(kpi + 1, kpj) - curr.at<float>(kpi - 1, kpj)) * firstDerivScale +
		dj * (curr.at<float>(kpi, kpj + 1) - curr.at<float>(kpi, kpj - 1)) * firstDerivScale +
		ds * (upper.at<float>(kpi, kpj) - lower.at<float>(kpi, kpj)) * firstDerivScale);
	// Filter by contrast threshold
	if (abs(ct) * (float)nLayers < contrastThres)
		return false;

	// Filter maximum along edges
	if (!elimEdgeResp(kpi, kpj, kps, otv))
		return false;

	// Set keypoint values
	kp.pt.x = ((float)kpj - dj) * pow(2.0, otv);
	kp.pt.y = ((float)kpi - di) * pow(2.0, otv);
	// Assign value to each of 3 bytes of the octave, by conventions of OpenCV
	kp.octave = (otv & 0xff) + ((kps & 0xff) << 8) + (((int)round((0.5 - ds) * 255)) << 16);
	kp.size = initSigma * pow(2.0, ((float)kps - ds) / (float)nLayers) * pow(2.0, otv) * 2.0;
	kp.response = abs(ct);
	return true;
}

// Calculate keypoint orientation
float SiftDOG::calcOrientation(int kpi, int kpj, int kps, int otv, int radius, float sigma, vector<float>& hist) {
	Mat& curr = dPyr[otv][kps];
	float weightFactor = -0.5 / (sigma * sigma);
	int area = (2 * radius + 1) * (2 * radius + 1);

	vector<float> M(area, 0.0);
	vector<float> W(area, 0.0);
	vector<float> O(area, 0.0);
	vector<float> H(SIFT_NUM_HIST_BINS, 0.0);
	int count = 0;

	// Accumlate orientation magnitude to correspoding bins
	for (int i = max(1, kpi - radius); i <= min(curr.rows - 2, kpi + radius); ++i) {
		for (int j = max(1, kpj - radius); j <= min(curr.cols - 2, kpj + radius); ++j) {
			float dx = curr.at<float>(i, j + 1) - curr.at<float>(i, j - 1);
			float dy = curr.at<float>(i - 1, j) - curr.at<float>(i + 1, j);
			M[count] = sqrt(dx * dx + dy * dy);
			W[count] = exp((float)((i - kpi) * (i - kpi) + (j - kpj) * (j - kpj)) * weightFactor);
			O[count] = fastAtan2(dy, dx);
			++count;
		}
	}

	// Expand the histogram for next step (smoothing)
	vector<float> xpdH(SIFT_NUM_HIST_BINS + 4, 0.0);
	for (int k = 0; k < count; ++k) {
		int binIdx = (int)round(O[k] * (float)SIFT_NUM_HIST_BINS / 360.0);
		binIdx = (binIdx + SIFT_NUM_HIST_BINS) % SIFT_NUM_HIST_BINS;
		H[binIdx] += W[k] * M[k];
		xpdH[binIdx + 2] = H[binIdx];
	}
	xpdH[0] = H[SIFT_NUM_HIST_BINS - 2];
	xpdH[1] = H[SIFT_NUM_HIST_BINS - 1];
	xpdH[SIFT_NUM_HIST_BINS + 2] = H[0];
	xpdH[SIFT_NUM_HIST_BINS + 3] = H[1];

	// Smooth histogram
	for (int binIdx = 0; binIdx < SIFT_NUM_HIST_BINS; ++binIdx) {
		hist[binIdx] = (xpdH[binIdx] + xpdH[binIdx + 4]) * 1.0 / 16.0 + (xpdH[binIdx + 1] + xpdH[binIdx + 3]) * 4.0 / 16.0 + xpdH[binIdx + 2] * 6.0 / 16.0;
	}
	float maxH = hist[0];
	// Find the main orientation (biggest magnitude)
	for (int binIdx = 1; binIdx < hist.size(); ++binIdx)
		maxH = max(maxH, hist[binIdx]);

	return maxH;
}

// Find local extrema function for multi-threading
void SiftDOG::findLocalExtremaTask(int otv) {
	cout << "[+++ TASK +++] findLocalExtremaTask, otv = " << otv << endl;
	keyPts[otv].resize(0);
	Octave& currOtv = dPyr[otv];
	int m = currOtv[0].rows, n = currOtv[0].cols;
	float thres = floor(0.5 * contrastThres / nLayers * 255.0);
	for (int l = 1; l < currOtv.size() - 1; ++l) {
		for (int i = border; i < m - border; ++i) {
			for (int j = border; j < n - border; ++j) {
				if (abs(currOtv[l].at<float>(i, j)) <= thres)
					continue;
				bool isMinimum = true, isMaximum = true;
				// Check if the pixel is 3x3x3 extrema
				for (int ll = l - 1; (isMinimum || isMaximum) && ll <= l + 1; ++ll) {
					for (int ii = i - 1; (isMinimum || isMaximum) && ii <= i + 1; ++ii) {
						for (int jj = j - 1; (isMinimum || isMaximum) && jj <= j + 1; ++jj) {
							if (ll == l && ii == i && jj == j)
								continue;
							if (currOtv[ll].at<float>(ii, jj) >= currOtv[l].at<float>(i, j))
								isMaximum = false;
							if (currOtv[ll].at<float>(ii, jj) <= currOtv[l].at<float>(i, j))
								isMinimum = false;
						}
					}
				}
				// If pixel is extrema
				if (isMinimum || isMaximum) {
					KeyPoint kp;
					int kpi = i, kpj = j, kps = l;
					// Localize extrema
					if (!localizeExtremum(kpi, kpj, kps, otv, kp))
						continue;
					// Find main orientation
					float scaleOtv = kp.size * 0.5 / pow(2.0, otv);
					vector<float> hist(SIFT_NUM_HIST_BINS, 0.0);
					float maxOri = calcOrientation(
						kpi, kpj, kps, otv,
						(int)round(SIFT_ORI_RADIUS * scaleOtv),
						SIFT_ORI_SIG_FCTR * (float)scaleOtv,
						hist
					);
					float magnitudeThres = maxOri * SIFT_ORI_PEAK_RATIO;
					// Assign orientation to extrema/keypoint
					for (int binIdx = 0; binIdx < SIFT_NUM_HIST_BINS; ++binIdx) {
						int left = (binIdx > 0) ? binIdx - 1 : SIFT_NUM_HIST_BINS - 1;
						int right = (binIdx < SIFT_NUM_HIST_BINS - 1) ? binIdx + 1 : 0;
						if (hist[binIdx] > magnitudeThres && hist[binIdx] > hist[left] && hist[binIdx] > hist[right]) {
							float bin = (float)binIdx + 0.5 * (hist[left] - hist[right]) / (hist[left] - 2 * hist[binIdx] + hist[right]);
							if (bin < 0.0)
								bin = bin + (float)SIFT_NUM_HIST_BINS;
							else if (bin >= (float)SIFT_NUM_HIST_BINS)
								bin = bin - (float)SIFT_NUM_HIST_BINS;
							kp.angle = 360.0 - (float)((360.0 / (float)SIFT_NUM_HIST_BINS) * bin);
							if (abs(kp.angle - 360.0) < FLT_EPSILON)
								kp.angle = 0.0;
							keyPts[otv].push_back(kp);
						}
					}
				}
			}
		}
	}
	cout << "[--- TASK ---] findLocalExtremaTask, otv = " << otv << endl;
}

// Find local extrema
void SiftDOG::findLocalExtrema() {
	allKeyPts.resize(0);
	keyPts.resize(nOctaves);
	{
		// Multi-threading initialization
		ThreadPool tp(-1);
		for (int o = 0; o < nOctaves; ++o)
			tp.enqueue([=](int otv) { findLocalExtremaTask(otv); }, o);
	}
	// Gather keypoints found by each thread task into a vector
	for (int o = 0; o < nOctaves; ++o) {
		allKeyPts.insert(allKeyPts.end(), keyPts[o].begin(), keyPts[o].end());
	}
}

// Remove duplicate keypoints (may occur after localizing keypoints)
void SiftDOG::sortAndRemoveDuplicateKeypoints() {
	if (allKeyPts.size() <= 1)
		return;

	sort(allKeyPts.begin(), allKeyPts.end(), [](const KeyPoint& kp1, const KeyPoint& kp2) {
		if (kp1.pt.x != kp2.pt.x) return kp1.pt.x < kp2.pt.x;
		if (kp1.pt.y != kp2.pt.y) return kp1.pt.y < kp2.pt.y;
		if (kp1.size != kp2.size) return kp1.size > kp2.size;
		if (kp1.angle != kp2.angle) return kp1.angle < kp2.angle;
		if (kp1.response != kp2.response) return kp1.response > kp2.response;
		if (kp1.octave != kp2.octave) return kp1.octave > kp2.octave;
		return kp1.class_id > kp2.class_id;
	});

	vector<KeyPoint> temp;
	for (int i = 0; i < allKeyPts.size(); ) {
		temp.push_back(allKeyPts[i]);
		const KeyPoint& kp = allKeyPts[i];
		while (
			++i < allKeyPts.size() &&
			allKeyPts[i].pt.x == kp.pt.x &&
			allKeyPts[i].pt.y == kp.pt.y &&
			allKeyPts[i].size == kp.size &&
			allKeyPts[i].angle == kp.angle
		) {}
	}

	allKeyPts = temp;
}

// Scale keypoints to original coords and size
void SiftDOG::denormalize() {
	for (KeyPoint& kp : allKeyPts) {
		// We multiply by 1/2 since the image [0][0] of pyramid is double in size
		kp.pt *= 0.5;
		kp.size *= 0.5;
		kp.octave = (kp.octave & ~255) | ((kp.octave - 1) & 255);
	}
}

// Calculate SIFT descriptor of a keypoint
void SiftDOG::calcDescriptor(Mat& curr, Point2f ptd, float ori, float scale, int kpIdx) {
	Point pt((int)round(ptd.x), (int)round(ptd.y));
	float cos_t = cos(ori * M_PI / 180.0);
	float sin_t = sin(ori * M_PI / 180.0);
	float binsPerRad = (float)SIFT_DESCR_HIST_BINS / 360.0;
	float expScale = -1.0 / ((float)SIFT_DESCR_WIDTH * (float)SIFT_DESCR_WIDTH * 0.5);
	float histWidth = SIFT_DESCR_SCL_FTR * scale;
	int radius = (int)round(histWidth * sqrt(2) * (SIFT_DESCR_WIDTH + 1) * 0.5);

	radius = min(radius, (int)sqrt((float)(curr.rows * curr.rows + curr.cols * curr.cols)));
	cos_t = cos_t / histWidth;
	sin_t = sin_t / histWidth;

	vector<float> M((2 * radius + 1) * (2 * radius + 1), 0.0);
	vector<float> W((2 * radius + 1) * (2 * radius + 1), 0.0);
	vector<float> O((2 * radius + 1) * (2 * radius + 1), 0.0);
	vector<float> XBin((2 * radius + 1) * (2 * radius + 1), 0.0);
	vector<float> YBin((2 * radius + 1) * (2 * radius + 1), 0.0);
	vector<float> rawRes((2 * radius + 1) * (2 * radius + 1), 0.0);

	vector<vector<vector<float>>> hist(
		SIFT_DESCR_WIDTH + 2,
		vector<vector<float>>(
			SIFT_DESCR_WIDTH + 2,
			vector<float>(
				SIFT_DESCR_HIST_BINS + 2,
				0.0
			)
		)
	);

	int count = 0;
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			float rotX = j * cos_t - i * sin_t;
			float rotY = j * sin_t + i * cos_t;
			float binX = rotX + SIFT_DESCR_WIDTH / 2 - 0.5;
			float binY = rotY + SIFT_DESCR_WIDTH / 2 - 0.5;
			int x = pt.x + j, y = pt.y + i;
			if (binX > -1.0 && binX < (float)SIFT_DESCR_WIDTH && binY > -1.0 && binY < (float)SIFT_DESCR_WIDTH && x > 0 && x < curr.cols - 1 && y > 0 && y < curr.rows - 1) {
				float dx = curr.at<float>(y, x + 1) - curr.at<float>(y, x - 1);
				float dy = curr.at<float>(y - 1, x) - curr.at<float>(y + 1, x);
				XBin[count] = binX;
				YBin[count] = binY;
				M[count] = sqrt(dx * dx + dy * dy);
				W[count] = exp((rotX * rotX + rotY * rotY) * expScale);
				O[count] = fastAtan2(dy, dx);
				++count;
			}
		}
	}

	for (int k = 0; k < count; ++k) {
		float cbin = XBin[k], rbin = YBin[k];
		float obin = (O[k] - ori) * binsPerRad;
		float mag = M[k] * W[k];

		int c0 = (int)floor(cbin);
		int r0 = (int)floor(rbin);
		int o0 = (int)floor(obin);
		cbin -= (float)c0;
		rbin -= (float)r0;
		obin -= (float)o0;

		if (o0 < 0) {
			o0 += SIFT_DESCR_HIST_BINS;
		}
		if (o0 >= SIFT_DESCR_HIST_BINS) {
			o0 -= SIFT_DESCR_HIST_BINS;
		}

		float v_r1 = mag * rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

		hist[r0 + 1][c0 + 1][o0] += v_rco000;
		hist[r0 + 1][c0 + 1][o0 + 1] += v_rco001;
		hist[r0 + 1][c0 + 2][o0] += v_rco010;
		hist[r0 + 1][c0 + 2][o0 + 1] += v_rco011;
		hist[r0 + 2][c0 + 1][o0] += v_rco100;
		hist[r0 + 2][c0 + 1][o0 + 1] += v_rco101;
		hist[r0 + 2][c0 + 2][o0] += v_rco110;
		hist[r0 + 2][c0 + 2][o0 + 1] += v_rco111;
	}

	for (int i = 0; i < SIFT_DESCR_WIDTH; ++i) {
		for (int j = 0; j < SIFT_DESCR_WIDTH; ++j) {
			hist[i + 1][j + 1][0] += hist[i + 1][j + 1][SIFT_DESCR_HIST_BINS];
			hist[i + 1][j + 1][1] += hist[i + 1][j + 1][SIFT_DESCR_HIST_BINS + 1];
			for (int k = 0; k < SIFT_DESCR_HIST_BINS; ++k)
				rawRes[(i * SIFT_DESCR_WIDTH + j) * SIFT_DESCR_HIST_BINS + k] = hist[i + 1][j + 1][k];
		}
	}

	float norm2 = 0.0;
	for (int k = 0; k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++k)
		norm2 += rawRes[k] * rawRes[k];
	float thres = sqrt(norm2) * SIFT_DESCR_MAG_THR;

	norm2 = 0.0;
	for (int k = 0; k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++k) {
		float val = min(rawRes[k], thres);
		rawRes[k] = val;
		norm2 += val * val;
	}
	norm2 = SIFT_INT_DESCR_FCTR / max(sqrt(norm2), FLT_EPSILON);

	float norm1 = 0.0;
	for (int k = 0; k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++k) {
		rawRes[k] *= norm2;
		norm1 += rawRes[k];
	}
	norm1 = 1.0 / max(norm1, FLT_EPSILON);

	for (int k = 0; k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; ++k) {
		descriptors.at<float>(kpIdx, k) = sqrt(rawRes[k] * norm1);
	}
}

// Caculate SIFT descriptors of keypoints
void SiftDOG::calcDescriptors() {
	descriptors = Mat::zeros(allKeyPts.size(), SIFT_DESCR_SIZE, CV_32F);
	for (int k = 0; k < allKeyPts.size(); ++k) {
		KeyPoint& kp = allKeyPts[k];
		int otv = kp.octave & 255;
		int layer = (kp.octave >> 8) & 255;
		otv = (otv < 128) ? otv : (-128 | otv);
		float scale = (otv >= 0) ? 1.0 / pow(2.0, otv) : pow(2.0, -otv);
		float size = kp.size * scale;
		float angle = 360.0 - kp.angle;
		if (abs(angle - 360.0) < FLT_EPSILON)
			angle = 0.0;

		Mat& curr = gPyr[otv + 1][layer];
		calcDescriptor(curr, Point2f(kp.pt.x * scale, kp.pt.y * scale), angle, size * 0.5, k);
	}
}

// Do detecte features by DOG algorithm
void SiftDOG::doDetectDOG() {
	init();
	cout << "init() complete" << endl;
	genSigmas();
	cout << "genSigmas() complete" << endl;
	createGaussianPyramid();
	cout << "createGaussianPyramid() complete" << endl;
	createDOGPyramid();
	cout << "createDOGPyramid() complete" << endl;
	findLocalExtrema();
	cout << "findLocalExtrema() complete" << endl;
	cout << allKeyPts.size() << endl;
	sortAndRemoveDuplicateKeypoints();
	cout << "sortAndRemoveDuplicateKeypoints() complete" << endl;
	denormalize();
	cout << "denormalize() complete" << endl;
	calcDescriptors();
	cout << "calcDescriptors() complete" << endl;
	cout << "Number of keypoints found: " << allKeyPts.size() << endl;
	drawKeypoints(res.clone(), allKeyPts, res, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// Return the result image
Mat SiftDOG::getResMat() {
	return res;
}

// Return the keypoints found
vector<KeyPoint> SiftDOG::getKeyPoints() {
	return allKeyPts;
}

// Return the descriptor matrix
cv::Mat SiftDOG::getDescriptors() {
	return descriptors;
}

// SIFT-DOG constructor
SiftDOG::SiftDOG(cv::Mat img, int nOctaves_, int nLayers_, float initSigma_, float constrastThres_, float edgeThres_, int nIters_, int border_, float prepSigma_) {
	this->orgImg = img;
	this->mat = img.clone();
	Mat imgAsInt;
	img.convertTo(imgAsInt, CV_8U);
	cvtColor(imgAsInt, this->res, COLOR_GRAY2BGR);
	orgHeight = img.rows;
	orgWidth = img.cols;

	nOctaves = nOctaves_;
	nLayers = nLayers_;
	initSigma = initSigma_;
	step = pow(2.0, 1.0 / (float)nLayers);
	contrastThres = constrastThres_;
	edgeThres = edgeThres_;
	nIters = nIters_;
	border = border_;
	prepSigma = prepSigma_;
}

// Boilerplate code for initializing algorithm input and parameters
Mat SiftDOG::detectDOG(Mat img, unordered_map<string, float> params, vector<KeyPoint>& kps, Mat& descriptors) {
	float initSigma_ = (params.find("init_sigma") != params.end()) ? params["init_sigma"] : 1.6;
	int nOctaves_ = (params.find("num_octaves") != params.end()) ? (int)params["num_octaves"] : 4;
	int nLayer_ = (params.find("num_layers") != params.end()) ? (int)params["num_layers"] : 3;
	float k_ = (params.find("k") != params.end()) ? params["k"] : pow(2.0, 0.25);
	float constrastThres_ = (params.find("constrast_thres") != params.end()) ? params["contrast_thres"] : 0.04;
	float edgeThres_ = (params.find("edge_thres") != params.end()) ? params["edge_thres"] : 10.0;
	int nIters_ = (params.find("localize_num_iters") != params.end()) ? (int)params["localize_num_iters"] : 5;
	int border_ = (params.find("border") != params.end()) ? (int)params["border"] : 5;
	int blurSize_ = (params.find("blur_ksize") != params.end()) ? (int)params["blur_ksize"] : 3;
	float prepSigma_ = (params.find("blur_sigma") != params.end()) ? params["blur_sigma"] : 0.5;

	Mat src;
	if (img.channels() == 1) {
		img.convertTo(src, CV_32F);
	}
	else if (img.channels() == 3) {
		Mat gray;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		gray.convertTo(src, CV_32F);
	}

	if (blurSize_ > 0 || prepSigma_ > 0.0) {
		src = gaussianBlur(src, blurSize_, prepSigma_);
	}

	SiftDOG siftDetector = SiftDOG(src, nOctaves_, nLayer_, initSigma_, constrastThres_, edgeThres_, nIters_, border_, prepSigma_);

	siftDetector.doDetectDOG();
	kps = siftDetector.getKeyPoints();
	descriptors = siftDetector.getDescriptors();
	return siftDetector.getResMat();
}