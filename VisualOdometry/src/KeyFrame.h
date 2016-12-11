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
public:
	Frame* frame = nullptr;
	KeyFrame* prev_kf = nullptr;
	//vector<Mat> points3D;
	vector<Frame*> frameVec;
	Mat T;
	Mat M1, M2;
	Mat point3D;
	Mat point3Dglobal;
	bool reconstructionDone;
	int timestamp;

public:
	KeyFrame(int, Frame* frame);
	Frame* getFrame();
	void registerKeyFrame();
	Mat reconstructFromPrevKF(vector<vector<Point2f>> pts);
	Mat reconstructFromPrevFrame();
	Mat stereoReconstruct();
	Mat get3DPoints();
	Mat get3DPointsGlobal();
	Mat getProjectionMat();
	Mat getProjectionMat2();
	Mat getNew3DPoints();
	void setPoseKF(Mat _T);
	vector<Frame*>& getFramesList();
	vector<Mat> getCommon3DPoints();
	void setPrevKeyFrame(KeyFrame* prev_kf);
	void setGlobalTransformation(Mat T);
	void addFrames(Frame* frame);
	Mat getPoseKF();
	void updatePoseKF();
	KeyFrame* getPrevKeyFrame();
	bool has3DPoints();
	virtual ~KeyFrame();
};

#endif /* KEYFRAME_H_ */
