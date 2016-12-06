/*
 * KeyFrame.h
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
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
	KeyFrame* prev_kf;
	//vector<Mat> points3D;
	vector<Frame*> frameVec;
	Mat T;
	Mat M1, M2;
	Mat point3D;
	Mat point3Dglobal;
    int timestamp;

public:
	KeyFrame(int, Frame* frame);
	Frame* getFrame();
	void registerKeyFrame();
	Mat stereoReconstruct();
	Mat get3DPoints();
	Mat get3DPointsGlobal();
	Mat getProjectionMat();
	vector<Mat> getCommon3DPoints();
	void setPrevKeyFrame(KeyFrame* prev_kf);
	void setGlobalTransformation(Mat T);
	void addFrames(Frame* frame);
	Mat getPoseKF();
	void updatePoseKF();
	KeyFrame* getPrevKeyFrame();
	virtual ~KeyFrame();

    void reconstructFromPrevKF(KeyFrame *prev_kf);
};

#endif /* KEYFRAME_H_ */
