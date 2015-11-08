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

typedef struct gVariables{
	/*These variables are utilized by keypoint detection and matching.*/
	int minHessian = 400; // Used for computing Keypoints. Smaller the number, less keypoints are shown.
	int max_dist = 0; // Max distance allowed between two points for "good match"
	int min_dist = 100; // Min distance allowed between two points for "good match"
	double multiplierDist = 2.0; // Multiplier for min dist

	int getMinHessian(){ return minHessian; }
	int getMax_Dist(){ return max_dist; }
	int getMin_Dist(){ return min_dist; }
	float getMultiplierDist() { return multiplierDist; }
};

int keypointDetection(string imageName1, string imageName2){
	gVariables g;
	Mat img_1 = imread(imageName1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread(imageName2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	if (img_1.empty())
	{
		printf("Unable to read first image. \n");
		return -1;
	}
	else if (img_2.empty())
	{
		printf("Unable to read second image. \n");
		return -1;
	}
	else{
		printf("Images loaded. \n");
	}

	// Initiate ORB Feature Detection.
	// STEP 1 -- Draw keypoints on each image.
	// Detector: ORB
	int minHessian = g.getMinHessian();
	Ptr<FeatureDetector> detector = ORB::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// Draw keypoints.
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1));
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1));

	// Show detected keypoints.
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	return 1;
}

int extractAndMatch(string imageName1, string imageName2){
	gVariables g;
	Mat img_1 = imread(imageName1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread(imageName2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	if (img_1.empty())
	{
		printf("Unable to read first image. \n");
		return -1;
	}
	else if (img_2.empty())
	{
		printf("Unable to read second image. \n");
		return -1;
	}
	else{
		printf("Images loaded. \n");
	}

	// Initiate ORB Feature Detection.
	// STEP 1 -- Draw keypoints on each image.
	// Detector: ORB
	int minHessian = g.getMinHessian();
	Ptr<FeatureDetector> detector = ORB::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// Draw keypoints.
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1));
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1));

	// Step 2: Calculate descriptors (feature vectors)
	// Extractor: ORB
	Ptr<DescriptorExtractor> extractor = SurfDescriptorExtractor::create();
	Mat descriptors_1, descriptors_2;
	vector<Mat> descriptors;

	extractor->compute(img_1, keypoints_1, descriptors_1);
	extractor->compute(img_2, keypoints_2, descriptors_2);
	descriptors_1.convertTo(descriptors_1, CV_32F); // Must convert each Desciptor to CV_32F format for FlannBasedMatcher
	descriptors_2.convertTo(descriptors_2, CV_32F);


	// Step 3: Matching descriptor vectors using FLANN Matching.
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	//double max_dist = 0; double min_dist = 30;
	double max_dist = g.getMax_Dist();
	double min_dist = g.getMin_Dist();

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	float multi = g.getMultiplierDist();
	printf("-- Multiplier: %f \n", multi);
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(multi * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	imshow("Good Matches", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}
}