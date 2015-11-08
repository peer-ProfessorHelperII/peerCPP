#define _CRT_SECURE_NO_DEPRECATE
#include <cstdlib>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\xfeatures2d.hpp> // OPENCV_CONTRIB
#include "detect_extract.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main( int argc, char** argv){
	cout << "You are now using OpenCV Version: " << CV_VERSION << endl;
	cout << "Please select an option:" << endl;
	cout << "1. View only keypoints" << endl;
	cout << "2. Perform matching" << endl;

	string imageName1;
	string imageName2;
	string inputImgName1, inputImgName2;

	// Read images here.
	int inputOption;
	cin >> inputOption;
	switch (inputOption){
	case 1:
		cout << "Please type the name of the photos. \n";
		cout << "Name of first image? \n";
		cin >> inputImgName1;
		imageName1 = inputImgName1;
		cout << "Name of second image? \n";
		cin >> inputImgName2;
		imageName2 = inputImgName2;
		cout << "Performing keypoint detection... \n";
		keypointDetection(imageName1, imageName2);
		break;
	case 2:
		cout << "Please type the name of the photos. \n";
		cout << "Name of first image? \n";
		cin >> inputImgName1;
		imageName1 = inputImgName1;
		cout << "Name of second image? \n";
		cin >> inputImgName2;
		imageName2 = inputImgName2;
		cout << "Performing keypoint matching... \n";
		extractAndMatch(imageName1, imageName2);
		break;
	default:
		cout << "Invalid command! Aborting... \n";
		return -1;
		break;
	}

	waitKey(0);
	return 0;
}