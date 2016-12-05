/*
 * Map.h
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi
 */

#ifndef MAP_H_
#define MAP_H_

#include <iostream>
#include <vector>
#include "KeyFrame.h"

using namespace std;

class Map {
	vector<KeyFrame *> mapkeyFrames;
	Mat getTfromCommon3D(vector<Mat> _points3d);
	viz::Viz3d myWindow;
public:
	Map();
	int getNumKeyFrames();
	void insertKeyFrame(KeyFrame* kf);
	void registerCurrentKeyFrame();
	void renderCurrentKF();
	void renderCurrentCamera(viz::WCameraPosition camPos, Affine3d cam_pose);
	virtual ~Map();
};

#endif /* MAP_H_ */
