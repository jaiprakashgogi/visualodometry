/*
 * Frame.h
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi
 */

#ifndef FRAME_H_
#define FRAME_H_

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
#include "KeyFrame.h"

using namespace std;
using namespace cv;

class Frame {
	KeyFrame* kf;
	Mat& frame;
	vector<KeyPoint> kpts;
	Mat& desc;
	Mat F; //Fundamental Matrix
	Mat T;
	int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);
	void symmetryTest(const vector<vector<DMatch> >& matches1,
			const vector<vector<DMatch> >& matches2,
			vector<DMatch>& symMatches);
	Mat ransacTest(const vector<DMatch>& matches,
			const vector<KeyPoint>& keypoints1,
			const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches,
			vector<Point2f>& points1, vector<Point2f>& points2);
public:
	Frame(string filename);
	Mat& getFrame();
	void extractFeatures();
	void matchFeatures();
	vector<KeyPoint>* getKeyPoints();
	Mat& getDesc();
	Mat& getPose();
	bool isKeyFrame();
	virtual ~Frame();
};

#endif /* FRAME_H_ */
