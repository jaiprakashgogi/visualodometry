/*
 * KeyFrame.h
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi
 */

#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "Frame.h"

using namespace std;
using namespace cv;

class Frame;

class KeyFrame {
	Frame* frame;
	//KeyFrame* prev_keyFrame;
	//vector<Mat> points3D;
	Mat T_prev;
	Mat T;
	Mat M1, M2;
	Mat point3D;
public:
	KeyFrame(Frame* frame);
	Frame* getFrame();
	void registerKeyFrame();
	Mat stereoReconstruct();
	Mat get3DPoints();
	virtual ~KeyFrame();
};

#endif /* KEYFRAME_H_ */
