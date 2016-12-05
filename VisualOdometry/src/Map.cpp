/*
 * Map.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "Map.h"

Map::Map() {
	// TODO Auto-generated constructor stub
	// Initialize Viz
	myWindow = viz::Viz3d("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

}

Map::~Map() {
	// TODO Auto-generated destructor stub
}

int Map::getNumKeyFrames() {
	return mapkeyFrames.size();
}

void Map::insertKeyFrame(KeyFrame* kf) {
	mapkeyFrames.push_back(kf);
}

void Map::registerCurrentKeyFrame() {
	cout << __func__ << ": " << mapkeyFrames.size() << endl;
	if(mapkeyFrames.size() < 2) {
		cout << __func__ << ": Only one keyframe" << endl;
		return;
	}

	int curr_id = mapkeyFrames.size() - 1;
	KeyFrame* curr_kf = mapkeyFrames.at(curr_id);

	//Find common points between the two keyframes
	vector<Mat> _points3d = curr_kf->getCommon3DPoints();
	Mat T = getTfromCommon3D(_points3d);
	curr_kf->setGlobalTransformation(T);
	return;
}


Mat Map::getTfromCommon3D(vector<Mat> _points3d) {
	//int nIter = 100;
	Mat pts1 = _points3d.at(0);
	Mat pts2 = _points3d.at(1);


	Mat pt1_mean(1, pts1.cols, pts1.type());
	Mat pt2_mean(1, pts2.cols, pts2.type());
	reduce(pts1, pt1_mean, 0, CV_REDUCE_AVG);
	reduce(pts2, pt2_mean, 0, CV_REDUCE_AVG);

	Mat pt1_mean_repeat(pts1.rows, pts1.cols, pts1.type());
	Mat pt2_mean_repeat(pts2.rows, pts1.cols, pts2.type());
	repeat(pt1_mean, pts1.rows, 1, pt1_mean_repeat);
	repeat(pt2_mean, pts2.rows, 1, pt2_mean_repeat);

	Mat pt1_new(pts1.rows, pts1.cols, pts1.type());
	Mat pt2_new(pts2.rows, pts2.cols, pts2.type());
	subtract(pts1, pt1_mean_repeat, pt1_new);
	subtract(pts2, pt2_mean_repeat, pt2_new);

	Mat S = pt1_new.t() * pt2_new;
	SVD svd(S);
	Mat R = svd.vt.t() * svd.u.t();
	Mat _t = pt2_mean.t() - R * pt1_mean.t();
	Mat T = Mat::eye(4, 4, CV_32F);
	T(Range(0, 3), Range(0, 3)) = R * 1;
	T(Range(0, 3), Range(3, 4)) = _t * 1;
	//cout << __LINE__ << T << R << _t << endl;
	return T;
}

void Map::renderCurrentKF() {
	int current_id = mapkeyFrames.size() - 1;
	Mat points3D = mapkeyFrames.at(current_id)->get3DPoints();
	viz::WCloud cloud_widget(points3D, viz::Color::green());
	myWindow.showWidget("3D view", cloud_widget);
	myWindow.spinOnce(1, true);
}

void Map::renderCurrentCamera(viz::WCameraPosition camPos, Affine3d cam_pose) {
	myWindow.showWidget("CPW1", camPos, cam_pose);
	myWindow.spinOnce(1, true);
}
