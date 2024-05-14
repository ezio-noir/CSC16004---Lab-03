#include "algs.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// Parse the string of format: --params:param1=val1;params2=val2;... into algorithm parameters
void parseParams(string str, unordered_map<string, float>& params) {
	if (str.substr(0, 9).compare("--params:") == 0) {
		params.clear();
		string strToParse = str.substr(9);
		vector<string> tokens;

		// Loop through the string to parse pairs of param=val
		int offset = 0, semicolonPos;
		do {
			semicolonPos = strToParse.find(",", offset);
			tokens.push_back(strToParse.substr(offset, semicolonPos - offset));
			offset = semicolonPos + 1;
		} while (semicolonPos != strToParse.npos);

		// Parse each pair of param=val and store it into the unordered map
		string param;
		float value;
		for (const auto& token : tokens) {
			param = token.substr(0, token.find("="));
			value = stof(token.substr(param.length() + 1));
			params[param] = value;
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc < 3 || argc > 6) {
		return -1;
	}

	// Read original image into grayscale
	Mat img = imread(argv[1], IMREAD_ANYCOLOR);
	Mat res;
	// Display orignial image
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);

	// An unordered map to store parameter value for algorithm
	unordered_map<string, float> params;
	// Parse the `--params:param1=val1,...` if it is specified by user
	if (argc == 4 && strcmp(argv[3], "-m") && strcmp(argv[3], "--match"))
		parseParams(argv[3], params);
	else if (argc == 6)
		parseParams(argv[5], params);

	vector<KeyPoint> kps;
	Mat descriptors;
	if (strcmp(argv[2], "--harris") == 0 || strcmp(argv[2], "-h") == 0) {
		res = Harris::detectHarris(img, params);
	}
	else if (strcmp(argv[2], "--blob") == 0 || strcmp(argv[2], "-b") == 0) {
		res = Blob::detectBlob(img, params);
	}
	else if (strcmp(argv[2], "--dog") == 0 || strcmp(argv[2], "-d") == 0) {
		res = SiftDOG::detectDOG(img, params, kps, descriptors);
	}
	else if (strcmp(argv[3], "--match") == 0 || strcmp(argv[3], "-m") == 0) {
		Mat img1 = imread(argv[1], IMREAD_ANYCOLOR);
		Mat img2 = imread(argv[2], IMREAD_ANYCOLOR);
		int detector = stoi(argv[4]);
		double matchScore = KnnMatch::match(img1, img2, detector, res, params);
		cout << "Matches: " << matchScore << endl;
	}

	imwrite("result.png", res);
	namedWindow("Result", WINDOW_AUTOSIZE);
	imshow("Result", res);

	waitKey(0);

	return 0;
}




