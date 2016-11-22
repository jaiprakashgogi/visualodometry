/*
 * cmusfm.h
 *
 *  Created on: Feb 24, 2016
 *      Author: jaiprakashgogi
 */

#ifndef CMUSFM_H_
#define CMUSFM_H_
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;

#define DEBUG 0

class cmusfm {
public:
	vector<Point2f> points[2];
	Mat point_3d;
	vector<string> filenames;
	Mat K; // intrinsic matrix
	int no_images;
	int MAX_COUNT;
	vector<Scalar> color;

	cmusfm();
	virtual ~cmusfm();
	void readfiles(string prefix);
	Mat showKLT(int i);
	Mat find3D();
	void setIntrinsic(Mat K);
	Mat findM2(Mat E);
	Mat getCamerapose(int id);
};

#endif /* CMUSFM_H_ */
