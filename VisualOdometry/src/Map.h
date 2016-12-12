/*
 * Map.h
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#ifndef MAP_H_
#define MAP_H_

#include <iostream>
#include <vector>
#include "KeyFrame.h"

class KeyFrame;

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Map {
public:
	enum MODE { MONO = 0, STEREO };
    vector<Point3f> pt3d;
	vector<KeyFrame *> mapkeyFrames;
	Mat getTfromCommon3D(vector<Mat> _points3d);
	viz::Viz3d myWindow;
	int kf_count;
	int cam_count;
	MODE mode;
	uint32_t frame_counter = 0;

public:
	Map();
	int getNumKeyFrames();
	void insertKeyFrame(KeyFrame* kf);
	void registerCurrentKeyFrame();
	void incrementTimestamp();
	void renderCurrentKF();
	void renderCurrentCamera(viz::WCameraPosition camPos, Affine3d cam_pose);
	void renderPointCloud(Mat points3D);
	void renderKFCameras();
	void setViewerPose(Affine3d viewer_pose);
	void setMode(MODE);
    void saveScreenshot();
	Map::MODE getMode();
	virtual ~Map();
};

#endif /* MAP_H_ */
